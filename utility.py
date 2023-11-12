import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

class Captioner():
    def __init__(self, model_path_or_link) -> None:
        print("Load model ...")
        self.processor = BlipProcessor.from_pretrained(model_path_or_link)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path_or_link).to(device)
        print("Model loaded!")
    
    def infer(self, image):
        inputs = self.processor(image, return_tensors="pt").to(device)
        out = self.model.generate(**inputs)

        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption

def read_image(path):
    raw_image = Image.open(path).convert('RGB')

    return raw_image

if __name__ == "__main__":
    pass
