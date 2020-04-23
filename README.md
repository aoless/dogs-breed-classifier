# Dog Breeds Classifier
Web application to recognize dogs based on their images

### Motivation
Nowadays, it seems that we have unlimited access to data. Especially when it comes to good quality photos. Each of us has a smartphone that we use to document scenes from our lives. One of the things we love to photograph are our pets.

One of the most popular uses of convolutional neural networks is object recognition in photos. The purpose is to conveniently segregate data, create amazing applications of augmented reality or intelligent analysis of monitoring recordings.

This application uses convolution neural network that can distinguish between 133 different breeds of dogs based on photos.
If you are big dogs fan and you keep a lot of photos on the computer you can use this code to segregate images according to breed or just use it to have fun by loading images of your friends and check which dog breed they resemble the most!


### Table of Contents

1. [Installation](#installation)
2. [Training](#training)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Create virtual envinroment 
`python -m venv "env_name"`

2. Install necessary libraries
`pip install -r requirements.txt`

3. Export path into env_name/bin/activate by writing on a top of a file

`export OLD_PYTHONPATH="$PYTHONPATH"`

`export PYTHONPATH="path_to_project/disaster-response-project/"`

4. Run the following command in the app's directory to run your web app
    `python run.py`

5. Go to http://0.0.0.0:3001/

## Training<a name="training"></a>
If you want to train model and play a little bit with data feel free to use notebook
attached in this repo. Follow the cells and you should not have the slightest problems with creating your own model :)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The MIT License
