import json
import os
import requests
from pathlib import Path


API_URL = os.getenv("EVAL_API_URL", "http://127.0.0.1:8000")
TRACES_DIR = Path("traces")


def load_traces():
    traces = []
    for file in sorted(TRACES_DIR.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            trace = json.load(f)
            trace["file"] = file.name
            traces.append(trace)
    return traces


def call_api(messages):
    response = requests.post(
        API_URL + "/chat",
        json={"messages": messages},
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def check_keywords(reply, recommendations, expected_keywords):
    full_text = reply.lower()
    for item in recommendations:
        full_text = full_text + " " + item.get("name", "").lower()
        full_text = full_text + " " + item.get("test_type", "").lower()

    matched = []
    missed = []
    for keyword in expected_keywords:
        if keyword.lower() in full_text:
            matched.append(keyword)
        else:
            missed.append(keyword)

    return matched, missed


def run_trace(trace):
    file_name = trace.get("file", "")
    description = trace.get("description", "")
    messages = trace.get("messages", [])
    expected_keywords = trace.get("expected_keywords", [])

    print("File        : " + file_name)
    print("Description : " + description)

    try:
        result = call_api(messages)
    except requests.exceptions.Timeout:
        print("Error       : Request timed out after 120 seconds.")
        print("-" * 60)
        return False
    except Exception as e:
        print("Error       : Could not reach API. " + str(e))
        print("-" * 60)
        return False

    reply = result.get("reply", "")
    recommendations = result.get("recommendations", [])
    end_of_conversation = result.get("end_of_conversation", False)

    print("Reply       : " + reply[:200])
    print("Count       : " + str(len(recommendations)) + " recommendations")
    print("Ended       : " + str(end_of_conversation))

    for i, item in enumerate(recommendations, start=1):
        name = item.get("name", "")
        test_type = item.get("test_type", "")
        url = item.get("url", "")
        print("  " + str(i) + ". " + name + " | " + test_type + " | " + url)

    matched, missed = check_keywords(reply, recommendations, expected_keywords)

    if matched:
        print("Matched     : " + ", ".join(matched))
    else:
        print("Matched     : none")

    if missed:
        print("Missed      : " + ", ".join(missed))
    else:
        print("Missed      : none")

    passed = len(recommendations) > 0 and len(missed) == 0

    if passed:
        print("Result      : PASS")
    else:
        print("Result      : FAIL")

    print("-" * 60)
    return passed


def run_all():
    print("SHL Assessment Agent Evaluation")
    print("API : " + API_URL)
    print("=" * 60)

    traces = load_traces()

    if not traces:
        print("No trace files found in the traces folder.")
        return

    total = len(traces)
    passed_count = 0

    for trace in traces:
        result = run_trace(trace)
        if result:
            passed_count = passed_count + 1

    failed_count = total - passed_count
    score = round((passed_count / total) * 100)

    print("=" * 60)
    print("Total  : " + str(total))
    print("Passed : " + str(passed_count))
    print("Failed : " + str(failed_count))
    print("Score  : " + str(score) + "%")


if __name__ == "__main__":
    run_all()