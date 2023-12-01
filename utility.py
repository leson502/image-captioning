import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models import blip_decoder, blip_vqa
image_size = 384

device = "cuda" if torch.cuda.is_available() else "cpu"

class Caption():
    def __init__(self) -> None:
        print("Load model ...")
        self.model = blip_decoder(pretrained="checkpoint/model_base_caption_capfilt_large.pth", image_size=image_size, vit='base').to(device)
    
    def generate(self, raw_image, max_length=30, min_length=10):

        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 

        image = transform(raw_image).unsqueeze(0).to(device)   

        self.model.eval()
        with torch.no_grad():
            caption = self.model.generate(image=image, sample=True, min_length=min_length, max_length=max_length)
        return caption[0]

class VQA():
    def __init__(self) -> None:
        print("Load model ...")
        self.model = blip_vqa(pretrained="checkpoint/model_base_vqa_capfilt_large.pth", image_size=image_size, vit='base').to(device)

    def generate(self, image, question):
        inputs = self.processor(image, question, return_tensors="pt").to(device)
        out = self.model.generate(**inputs)

        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption
    
def read_image(path):
    raw_image = Image.open(path).convert('RGB')
    
    return raw_image

if __name__ == "__main__":
    pass
