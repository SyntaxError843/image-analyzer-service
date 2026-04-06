from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
from PIL import Image
from io import BytesIO

from app.model import analyze_model
    
app = FastAPI()

class AnalyseImageRequest(BaseModel):
    image_urls: List[HttpUrl]
    conditional_text: str = ''

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
def analyse_image(request: AnalyseImageRequest):
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")

    results = []

    for url in request.image_urls:
        try:
            image = download_image(str(url))

            captions = analyze_model.caption_image(image, request.conditional_text)

            results.append({
                "url": url,
                "captions": captions
            })

        except Exception as e:
            results.append({
                "url": url,
                "error": str(e)
            })

    return {
        "model": "Salesforce/blip-image-captioning-base",
        "results": results
    }