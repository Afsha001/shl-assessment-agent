import os
import json
import logging
import traceback
from groq import Groq
from dotenv import load_dotenv

from app.retrieval import search
from app.catalog import CATALOG

load_dotenv()

logger = logging.getLogger(__name__)

CATALOG_URLS = {item["url"] for item in CATALOG}

MAX_TURNS = 8

ANCHOR_SLUGS = [
    "occupational-personality-questionnaire-opq32r",
    "shl-verify-interactive-g",
    "graduate-scenarios",
]

SYSTEM_PROMPT = """CRITICAL: Your response must be pure JSON only. No text before or after the JSON. No markdown. No backticks. Start your response with { and end with }. Any text outside the JSON object will break the system.

You are an SHL assessment recommender agent. Your job is to help hiring managers and recruiters find the right SHL assessments for their hiring needs.

STRICT RULES:
1. You ONLY discuss SHL assessments from the catalog. Refuse all other topics politely.
2. NEVER recommend an assessment that is not in the catalog context provided to you.
3. NEVER invent URLs. Only use URLs exactly as provided in the catalog context.
4. Every URL you return must come verbatim from the catalog context below.
5. Maximum 8 turns per conversation. If turn 8 is reached, provide your best recommendations and set end_of_conversation to true.

BEHAVIORAL RULES:
6. CLARIFY before recommending if the request is too vague. You need at minimum: job role, seniority level, and purpose. If any of these are missing, ask for them. Do not recommend on the first turn for vague queries.
7. If the request contains enough detail (role, level, purpose), recommend immediately without asking clarifying questions.
8. RECOMMEND 1 to 10 assessments once you have enough context. Use name, url, and test_type exactly from the catalog.
9. REFINE when user changes constraints. Update the existing shortlist, add or remove items, and explain what changed. Never restart from scratch.
10. COMPARE when asked about differences between assessments. Answer from catalog data only. Keep recommendations as empty list for compare turns.
11. REFUSE off-topic questions politely. Say you can only help with SHL assessments. Do not end the conversation.
12. Add OPQ32r as a default personality component for professional or senior roles unless user explicitly excludes it. Always flag it: I have included OPQ32r as a default personality measure, say the word if you prefer to drop it.
13. Set end_of_conversation to true ONLY when the user explicitly says they are done, for example: confirmed, that is all, perfect, locking it in. Never assume the conversation is over just because you gave recommendations.

OUTPUT FORMAT:
Always respond with this exact JSON structure and nothing else.

{
  "reply": "your conversational response here",
  "recommendations": [],
  "end_of_conversation": false
}

When recommending, recommendations is a list of 1 to 10 objects:
{"name": "...", "url": "...", "test_type": "..."}

When clarifying, comparing, or refusing, recommendations must be an empty list.
When user confirms they are done, set end_of_conversation to true and repeat the final recommendations."""


FALLBACK = {
    "reply": "I encountered an issue processing your request. Could you rephrase it?",
    "recommendations": [],
    "end_of_conversation": False,
}


def extract_json(text):
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response: " + text[:200])


def get_anchor_assessments():
    anchors = []
    for item in CATALOG:
        for slug in ANCHOR_SLUGS:
            if slug in item["url"]:
                anchors.append(item)
                break
    return anchors


def build_search_results(query):
    search_results = search(query, top_k=20)

    anchors = get_anchor_assessments()

    seen_urls = set()
    for item in anchors:
        seen_urls.add(item["url"])

    remaining = []
    for item in search_results:
        if item["url"] not in seen_urls:
            remaining.append(item)

    combined = anchors + remaining
    return combined[:20]


def build_catalog_context(candidates):
    lines = []
    for i, item in enumerate(candidates, start=1):
        levels = ", ".join(item.get("job_levels", []))
        duration = item.get("duration", "")
        description = item.get("description", "")[:200]
        line = (
            "[" + str(i) + "]"
            + " Name: " + item["name"]
            + " | URL: " + item["url"]
            + " | Type: " + item["test_type"]
            + " | Levels: " + levels
            + " | Duration: " + duration
            + " | Description: " + description
        )
        lines.append(line)
    return "\n".join(lines)


def get_last_shortlist(messages):
    for message in reversed(messages):
        if message.get("role") == "assistant":
            try:
                parsed = json.loads(message["content"])
                return parsed.get("recommendations", [])
            except Exception:
                return []
    return []


def count_turns(messages):
    return sum(1 for m in messages if m.get("role") == "user")


def clean_conversation(messages):
    cleaned = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            try:
                parsed = json.loads(content)
                content = parsed.get("reply", content)
            except Exception:
                pass
        cleaned.append({"role": role, "content": content})
    return cleaned


def filter_valid_urls(recommendations):
    valid = []
    for item in recommendations:
        if item.get("url") in CATALOG_URLS:
            valid.append(item)
    return valid


async def get_agent_response(messages):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY is not set in environment.")
            return FALLBACK

        turn_number = count_turns(messages)

        user_messages = [m for m in messages if m.get("role") == "user"]
        recent_query = " ".join(m["content"] for m in user_messages[-3:])

        candidates = build_search_results(recent_query)
        catalog_context = build_catalog_context(candidates)

        last_shortlist = get_last_shortlist(messages)

        turn_note = ""
        if turn_number >= MAX_TURNS:
            turn_note = (
                "\n\nTURN LIMIT: This is turn "
                + str(turn_number)
                + " of maximum 8. You must provide your best recommendations now and set end_of_conversation to true."
            )

        system_content = (
            SYSTEM_PROMPT
            + turn_note
            + "\n\nCATALOG CONTEXT (use ONLY these for recommendations):\n"
            + catalog_context
        )

        conversation = clean_conversation(messages)

        if last_shortlist:
            shortlist_reminder = (
                "CURRENT SHORTLIST: "
                + json.dumps(last_shortlist)
                + ". Update this list rather than starting fresh unless the user explicitly restarts."
            )
            conversation.append({"role": "user", "content": shortlist_reminder})

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_content}] + conversation,
            temperature=0.1,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        logger.info("Groq raw response: " + raw)

        parsed = extract_json(raw)

        reply = parsed.get("reply", "")
        recommendations = parsed.get("recommendations", [])
        end_of_conversation = parsed.get("end_of_conversation", False)

        if not isinstance(reply, str):
            reply = str(reply)
        if not isinstance(recommendations, list):
            recommendations = []
        if not isinstance(end_of_conversation, bool):
            end_of_conversation = False

        if turn_number >= MAX_TURNS:
            end_of_conversation = True

        recommendations = filter_valid_urls(recommendations)

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation,
        }

    except json.JSONDecodeError as e:
        logger.error("JSON parse error: " + str(e))
        logger.error(traceback.format_exc())
        print("AGENT ERROR: " + str(e))
        print(traceback.format_exc())
        return FALLBACK

    except Exception as e:
        logger.error("Agent error: " + str(e))
        logger.error(traceback.format_exc())
        print("AGENT ERROR: " + str(e))
        print(traceback.format_exc())
        return FALLBACK