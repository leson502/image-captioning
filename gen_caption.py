import json
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
device = "cuda" if torch.cuda.is_available() else "cpu"

from model import read_image
from tqdm import tqdm

if __name__ == '__main__':
    with open("data/captions_val2017.json", "r") as f:
        ann = json.load(f)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    data_root = "data/val2017"

    result = {}

    for image in tqdm(ann["images"]):
        path = os.path.join(data_root, image['file_name'])
        raw_img = read_image(path)

        inputs = processor(raw_img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        result[image['id']] = caption
    
    with open("output/result.json", "w") as f:
        json.dump(result, f)

    


    
    




    

    
        
        
