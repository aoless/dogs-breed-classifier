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


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


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