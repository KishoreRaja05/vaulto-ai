from typing import Optional

from fastapi import APIRouter, Header

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_agent import run_chat

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, authorization: Optional[str] = Header(None)):
    return run_chat(payload, authorization)


@router.get("/status")
async def chat_status():
    return {"status": "AI Chat service online"}
