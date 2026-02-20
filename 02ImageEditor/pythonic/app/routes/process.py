from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.services.image_processor import process_image
import io

router = APIRouter()

@router.post("/process")
async def process(
    file: UploadFile = File(...),
    filter_type: str = Form("none"),
    width: int = Form(None),
    height: int = Form(None),
    brightness: int = Form(None)
):
    contents = await file.read()

    processed = process_image(
        contents,
        filter_type,
        width,
        height,
        brightness
    )

    return StreamingResponse(
        io.BytesIO(processed),
        media_type="image/jpeg"
    )