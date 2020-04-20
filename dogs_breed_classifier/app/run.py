import json
import os
import urllib.request

import plotly
import pandas as pd

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from joblib import load
from plotly.graph_objs import Bar
from werkzeug.utils import secure_filename

from app import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
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
    return render_template(
        'go.html',
        filepath=filepath
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()