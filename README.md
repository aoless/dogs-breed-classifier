# Dog Breeds Classifier
Web application to recognize dogs based on their images

### Motivation
Nowadays, it seems that we have unlimited access to data. Especially when it comes to good quality photos. Each of us has a smartphone that we use to document scenes from our lives. One of the things we love to photograph are our pets.

One of the most popular uses of convolutional neural networks is object recognition in photos. The purpose is to conveniently segregate data, create amazing applications of augmented reality or intelligent analysis of monitoring recordings.

This application uses convolution neural network that can distinguish between 133 different breeds of dogs based on photos.
If you are big dogs fan and you keep a lot of photos on the computer you can use this code to segregate images according to breed or just use it to have fun by loading images of your friends and check which dog breed they resemble the most!


### Table of Contents

1. [Installation](#installation)
2. [Files description](#files)
3. [Training](#training)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Create virtual envinroment 
`python -m venv "env_name"`

2. Install necessary libraries
`pip install -r requirements.txt`

3. Export path into env_name/bin/activate by writing on a top of a file

`export OLD_PYTHONPATH="$PYTHONPATH"`

`export PYTHONPATH="path_to_project/dogs-breed-classifier/"`

4. Run the following command in the app's directory to run your web app
    `python run.py`

5. Go to http://0.0.0.0:3001/

## Files Descriptions <a name="files"></a>

| Module        | Module           | Explanation  |
| ------------- |:-------------:| -----:|
| app           | app.py         | setup flask application |
| app           | run.py         | run flask web application |
| app           | templates      | html templates |
| data          | bottleneck_features| pretrained ResNet50 features |
| data          | cascade_classifier| openCV model for recognizing human faces |
| data          | pretrained_models| pretrained tf 2.0 ResNet50 model |
| model        | resnet_model.py| create model and load weights |
| model        | classifier.py| module for classifying images |

## Training<a name="training"></a>
For training I've used data from: <a href="https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip" class="text-white">dataset</a>
If you want to train model and play a little bit with data feel free to use notebook
attached in this repo. Follow the cells and you should not have the slightest problems with creating your own model :)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
The MIT License
