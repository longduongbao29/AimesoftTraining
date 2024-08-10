from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

app = FastAPI()
from nlp.text_processor import TextProcessor


@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")

    processor = TextProcessor()
    processed_text = processor.process_text(text)

    # Tạo một stream để trả về file đã xử lý
    processed_file = io.BytesIO(processed_text.encode("utf-8"))
    return StreamingResponse(
        processed_file,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=processed_{file.filename}"
        },
    )
