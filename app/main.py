import logging
from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Agent")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "message": "SHL Assessment Agent is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        from app.agent import get_agent_response
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        result = await get_agent_response(messages)
        return ChatResponse(
            reply=result.get("reply", ""),
            recommendations=result.get("recommendations", []),
            end_of_conversation=result.get("end_of_conversation", False),
        )
    except Exception as e:
        logger.error("Error in chat endpoint: " + str(e))
        return ChatResponse(
            reply="Something went wrong. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )