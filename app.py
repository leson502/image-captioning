from flask import Flask, render_template, request, flash
from model import Caption, VQA, read_image
import os
app = Flask(__name__)

global_val = {
    "title": None,
    "mode": None,
    "model": None,
}

@app.route('/', methods=['GET'])
def main():
    return render_template('homepage.html')

@app.route('/caption', methods=['GET', 'POST'])
def load_caption():
    if global_val['mode'] != 'caption':
        global_val['mode'] = "caption"
        global_val["model"] = None
        global_val["model"] = Caption("Salesforce/blip-image-captioning-large")
        global_val["title"] = "Image captioning"
    return render_template("generate.html", title=global_val['title'])


@app.route('/vqa', methods=['GET', 'POST'])
def load_vqa():
    if global_val['mode'] != 'vqa':
        global_val['mode'] = "vqa"
        global_val['model'] = None
        global_val['model'] = VQA("Salesforce/blip-vqa-capfilt-large")
        global_val["title"] = "Visual answer questioning"
    return render_template("generate.html", title=global_val['title'])

@app.route('/result', methods=['GET', 'POST'])
def generate():
    imagefile = request.files['imagefile']
    text = request.form['condition-question']
    app_model = global_val['model']

    if imagefile.filename == '':
        return render_template('generate.html', title=global_val['title'], alert="You must provide image.")
    
    if global_val['mode'] == 'vqa' and text == '':
        return render_template('generate.html', title=global_val["title"], alert="You must provide question.")
    
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image = read_image(image_path)
    output = app_model.generate(image, text)

    if global_val['mode'] == 'caption':
        return render_template('generate.html', title=global_val['title'], caption=output, image_path="images/" + imagefile.filename)
    elif global_val['mode'] == 'vqa':
        return render_template('generate.html', title=global_val['title'], question=text, answer=output, image_path="images/" + imagefile.filename)

