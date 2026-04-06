from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch


class AnalyzerModel:
    def __init__(self):
        self.device = "cuda"

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct"  # fallback if 35B too heavy
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def analyse(self, image: Image.Image, prompt: str):
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=300)

        return self.processor.decode(output[0], skip_special_tokens=True)


analyzer_model = AnalyzerModel()