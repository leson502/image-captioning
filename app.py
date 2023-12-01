from flask import Flask, render_template, request, flash
from utility import Caption, VQA, read_image
import os
app = Flask(__name__)

global_val = {
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
        global_val["model"] = Caption()
    return render_template("caption.html",
                           min_len = 10,
                           max_len = 15)


@app.route('/vqa', methods=['GET', 'POST'])
def load_vqa():
    if global_val['mode'] != 'vqa':
        global_val['mode'] = "vqa"
        global_val['model'] = None
        global_val['model'] = VQA()
    return render_template("vqa.html")

@app.route('/vqa/result', methods=['GET', 'POST'])
def generate_vqa():
    imagefile = request.files['imagefile']
    question = request.form['question']
    app_model = global_val['model']
    
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image = read_image(image_path)
    output = app_model.generate(image, question)

    
    return render_template('vqa.html',
                           question=question,
                           answer=output,
                           image_path="images/" + imagefile.filename)

@app.route('/caption/result', methods=['GET', 'POST'])
def generate_caption():
    imagefile = request.files['imagefile']
    min_len = int(request.form['minLength'])
    max_len = int(request.form['maxLength'])
    print('here')

    app_model = global_val['model']
    
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image = read_image(image_path)
    output = app_model.generate(image, min_length=min_len, max_length=max_len)

    

    return render_template('caption.html', 
                            caption=output,
                            image_path = "images/" + imagefile.filename,
                            max_len=max_len,
                            min_len=min_len)
    

app.run()