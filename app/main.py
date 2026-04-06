from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
from PIL import Image
from io import BytesIO

from app.model import analyzer_model
    
app = FastAPI()

class AnalyseRequest(BaseModel):
    image_urls: List[HttpUrl]
    instructions: str = ''

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "image/*,*/*;q=0.8",
}

def download_image(url: str) -> Image.Image:
    response = requests.get(url, headers=HEADERS, timeout=10, stream=True)

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {url}"
        )

    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {url}"
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/analyse-image")
def analyse(request: AnalyseRequest):
    results = []

    for url in request.image_urls:
        try:
            image = download_image(str(url))

            prompt = request.instructions or "analyze the image and describe its content in detail"

            output = analyzer_model.analyse(image, prompt)

            results.append({
                "url": url,
                "raw_output": output
            })

        except Exception as e:
            results.append({
                "url": url,
                "error": str(e)
            })

    return {"results": results}