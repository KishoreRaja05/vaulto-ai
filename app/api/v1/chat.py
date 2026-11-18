from fastapi import APIRouter

# Initialize APIRouter instance
router = APIRouter()

@router.get("/status")
async def chat_status():
    return {"status": "AI Chat service online"}