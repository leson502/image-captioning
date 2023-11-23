from flask import Flask, render_template, request
from model import Captioner, read_image
import os
app = Flask(__name__)
model = Captioner("Salesforce/blip-image-captioning-large")

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image = read_image(image_path)
    caption = model.generate(image=image)
    print(image_path)
    return render_template('index.html', caption=caption, image_path="images/" + imagefile.filename)

