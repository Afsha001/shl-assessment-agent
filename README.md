<div align="center">
# SHL Assessment Recommender Agent

A conversational AI agent that helps hiring managers and recruiters find the right SHL assessments for their hiring needs. Built with FastAPI, Groq LLM, and TF-IDF retrieval over a catalog of 377 SHL assessments.

---

## Live API

Base URL: https://afsha001-shl-assessment-agent.hf.space

Health check: https://afsha001-shl-assessment-agent.hf.space/health

Interactive docs: https://afsha001-shl-assessment-agent.hf.space/docs
</div>

---

## What It Does

The agent runs a multi-turn conversation with the user. It asks clarifying questions when the request is too vague, recommends assessments once it has enough context, refines the shortlist when the user changes requirements, and ends the conversation only when the user explicitly confirms they are done.

Every recommendation comes directly from the SHL catalog. URLs are validated against the catalog before being returned. The agent never makes up an assessment or a URL.

---

## Project Structure

| File | Purpose |
|------|---------|
| app/main.py | FastAPI app with health and chat endpoints |
| app/schemas.py | Pydantic request and response models |
| app/catalog.py | Loads and cleans the SHL assessment catalog |
| app/retrieval.py | TF-IDF search over the catalog |
| app/agent.py | Groq LLM agent with prompt and response logic |
| data/catalog.json | SHL assessment catalog, 377 entries |
| traces/ | Five conversation trace files for evaluation |
| eval/run_traces.py | Evaluation script that scores agent responses |
| Dockerfile | Container setup for HuggingFace Spaces |
| requirements.txt | Python dependencies |
| .env.example | Environment variable template |
---

## API Contract

### GET /health

Returns a simple status response to confirm the service is running.

Response:

{
  "status": "ok"
}

### POST /chat

Accepts a full conversation history and returns the agent reply, a list of recommended assessments, and a flag indicating whether the conversation is complete.

Request:

{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior software engineer"}
  ]
}

Response:

{
  "reply": "For a senior software engineer I recommend the following assessments...",
  "recommendations": [
    {
      "name": "Core Java Advanced Level",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}

---

## How It Works

### Catalog Loading

The catalog is loaded from data/catalog.json at startup. Only entries with status ok are kept. Each entry is cleaned into a standard dict with fields for name, url, test type, job levels, languages, duration, remote, adaptive, and description. The test type field is built by mapping assessment category names to single letter abbreviations such as A for Ability and Aptitude, P for Personality and Behavior, and K for Knowledge and Skills.

### Retrieval

Search is powered by TF-IDF with bigrams over a corpus built from each assessment name, category keys, job levels, and description. When a query comes in, cosine similarity is computed against the full corpus and the top 20 matches are returned. Three anchor assessments, OPQ32r, SHL Verify Interactive G+, and Graduate Scenarios, are always prepended to the context so the LLM sees them on every turn regardless of the query.

### Agent

The agent is built on Groq using llama-3.3-70b-versatile. On each turn it receives a system prompt with behavioral rules and the catalog context for the top 20 retrieved assessments. The full conversation history is passed in so the agent can refine its shortlist across turns. After the LLM responds, every URL is validated against the catalog before the response is returned. Any URL not found in the catalog is removed.

The system prompt enforces these behaviors. The agent clarifies when the query is too vague. It recommends once it has enough context. It updates the existing shortlist rather than restarting when the user changes constraints. It refuses off-topic questions politely. It adds OPQ32r by default for senior roles and flags it to the user. It honors a maximum of 8 turns per conversation.

---

## Running Locally

Clone the repository and navigate into the project folder.

On Mac or Linux:

git clone https://github.com/Afsha001/shl-assessment-agent
cd shl-assessment-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

On Windows:

git clone https://github.com/Afsha001/shl-assessment-agent
cd shl-assessment-agent
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env

Open .env and add your Groq API key:

GROQ_API_KEY=your_actual_key_here

Place the catalog file at data/catalog.json, then start the server:

uvicorn app.main:app --reload

Visit http://127.0.0.1:8000/health to confirm the server is running and http://127.0.0.1:8000/docs for the interactive API documentation.

---

## Evaluation

The evaluation script runs five conversation traces against the live API and scores each one based on whether the agent returned at least one recommendation and matched all expected keywords.

Run against the local server:

python -m eval.run_traces

Run against the deployed HuggingFace endpoint:

set EVAL_API_URL=https://afsha001-shl-assessment-agent.hf.space
python -m eval.run_traces

The five traces cover these scenarios:

Trace 01: Hiring a senior software engineer with Java skills
Trace 02: Hiring entry level customer service agents for inbound calls
Trace 03: Hiring a senior manager for a leadership role
Trace 04: Hiring fresh graduates for a trainee program
Trace 05: Multi-turn conversation where the user removes personality assessments mid-conversation

Current score: 5 out of 5 traces passing, 100 percent.

---

## Deployment

The application is deployed on HuggingFace Spaces using Docker. The Dockerfile uses python:3.11-slim as the base image, installs dependencies from requirements.txt, copies the project files, and starts the server on port 7860 which is required by HuggingFace Spaces.

The GROQ_API_KEY is stored as a secret in the HuggingFace Space settings and is never committed to version control.

To push updates to the deployed service:

git push space main

---

## Issues Encountered and How They Were Fixed

Memory crash on Render free tier. The sentence-transformers model and FAISS index were loading at import time and consuming over 512MB before the server could start. We tried lazy loading first to defer the model until the first request. When that was not enough, we moved deployment to HuggingFace Spaces which provides 16GB of RAM on the free CPU tier.

LLM returning text outside the JSON. The Groq model occasionally added conversational text before or after the JSON object. We added a two-step extraction function that first tries a direct JSON parse, and if that fails, finds the outermost curly braces and extracts the content between them. We also added a strict instruction at the top of the system prompt reminding the model to return pure JSON only.

Hallucinated URLs in recommendations. The LLM sometimes generated URLs that looked plausible but did not exist in the catalog. We added a validation step that checks every recommended URL against the set of known catalog URLs and removes any item whose URL is not found before returning the response.

HuggingFace git authentication. Password authentication for HuggingFace git was deprecated. We resolved this by creating a write-access token in HuggingFace settings and embedding it in the remote URL.

Eval keyword mismatches. Early trace keywords were too specific and did not match the natural language the agent used in its replies. We reviewed the actual agent output for each trace and updated the keywords to match words that genuinely appeared in the responses.

---

## Dependencies

fastapi
uvicorn
pydantic
python-dotenv
groq
numpy
requests
scikit-learn
sentence-transformers
faiss-cpu

---

## Environment Variables

GROQ_API_KEY is your Groq API key and is required for the LLM to function. Add it to your .env file locally or as a secret in your HuggingFace Space settings.

