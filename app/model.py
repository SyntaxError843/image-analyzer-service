from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class AnalyzeModel:
    def __init__(self):
        # BLIP captioning model
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def caption_image(self, pil_image: Image.Image, conditional_text: str = "") -> str:
        # Conditional caption
        inputs = self.blip_processor(
            pil_image, conditional_text, return_tensors="pt"
        )

        out = self.blip_model.generate(**inputs)
        conditional_caption = self.blip_processor.decode(
            out[0], skip_special_tokens=True
        )

        return conditional_caption

# Singleton
analyze_model = AnalyzeModel()