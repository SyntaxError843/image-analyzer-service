from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import logging

from vllm import LLM, SamplingParams
    
app = FastAPI()
logger = logging.getLogger("uvicorn")

extraction_llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    dtype="bfloat16",
    max_model_len=16384,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.23,
)
logger.info("Qwen2.5 VL model loaded successfully")

class AnalyseRequest(BaseModel):
    image_url: HttpUrl
    instructions: str = ''

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/analyse-image")
def analyse(request: AnalyseRequest):
    try:
        if not request.image_url:
            raise HTTPException(
                status_code=400,
                detail=f"no image provided"
            )
        
        prompt = request.instructions or "analyze the image and describe its content in detail"

        outputs = extraction_llm.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": request.image_url},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            SamplingParams(temperature=0.0, max_tokens=2048),
        )

        raw = outputs[0].outputs[0].text.strip()

        return {
            "raw_output": raw
        }

    except Exception as e:
        return {
            "error": str(e)
        }
