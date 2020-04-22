import json
import os
import urllib.request
from joblib import load
from glob import glob

import plotly
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from joblib import load
from werkzeug.utils import secure_filename

from app import app
from dogs_breed_classifier.model import classifier
from dogs_breed_classifier.model import resnet_model

tf.get_logger().setLevel('INFO')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
CLASS_NUMS = 133

dog_names = load("../data/dog_names.pkl")

cascade_model_path = "../data/cascade_classifier/haarcascade_frontalface_alt.xml"

model = resnet_model.load_pretrained_model(
    CLASS_NUMS,
    "../data/pretrained_models/weights.best.resnet.hdf5",
    )


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # clean all uploaded files
    files = glob("static/uploaded/*.jpg")
    for f in files:
        os.remove(f)
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save("static/" + filepath)
            session['filepath'] = filepath
            return redirect(url_for('.go', filepath=filepath))
        else:
            return redirect(request.url)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    filepath = request.args['filepath']
    image_path_for_model = "static/" + filepath
    prediction = classifier.make_predictions(
        model,
        image_path_for_model,
        cascade_model_path,
        dog_names,
        )

    return render_template('go.html', filepath=filepath, prediction=prediction)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()