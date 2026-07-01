import logging
from fastapi import FastAPI

from app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Agent")

try:
    from app.agent import get_agent_response
    logger.info("Agent loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load agent: {e}")
    get_agent_response = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if get_agent_response is None:
        return ChatResponse(
            reply="Agent is currently unavailable. Please check server logs.",
            recommendations=[],
            end_of_conversation=False,
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = await get_agent_response(messages)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=result.get("recommendations", []),
        end_of_conversation=result.get("end_of_conversation", False),
    )