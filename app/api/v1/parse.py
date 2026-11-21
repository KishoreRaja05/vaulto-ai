import hmac

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile

from app.config import settings
from app.services.extractors import MAX_UPLOAD_BYTES, detect_file_type, parse_upload

router = APIRouter(tags=["Parse"])


def require_internal_parse_access(
    request: Request, x_internal_key: str | None = Header(None, alias="X-Internal-Key")
) -> None:
    if request.headers.get("origin"):
        raise HTTPException(status_code=403, detail="Browser access is not allowed.")
    if not settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=503, detail="Parse endpoint is not configured.")
    if not x_internal_key or not hmac.compare_digest(x_internal_key, settings.INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")


@router.post("/parse")
async def parse_file(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_internal_parse_access),
):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 5 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    detected_type = detect_file_type(data)
    result = parse_upload(data, detected_type)
    source_type = "image" if detected_type in {"png", "jpg"} else detected_type
    return {
        "source_type": source_type,
        "transactions": result["transactions"],
        "warnings": result.get("warnings", []),
        "used_llm": bool(result.get("used_llm", False)),
    }
