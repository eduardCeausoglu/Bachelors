import os
from flask import Flask, request, render_template_string
from scipy.misc import imread
import cv2
from keras.models import load_model
import numpy as np
from flask_classful import FlaskView, route
from sklearn.metrics import precision_score, recall_score, f1_score

class Server(FlaskView):
    def __init__(self):
        self.classes = {
            '0': 'maximum speed limit is 20',
            '1': 'maximum speed limit is 30',
            '2': 'maximum speed limit is 50',
            '3': 'maximum speed limit is 60',
            '4': 'maximum speed limit is 70',
            '5': 'maximum speed limit is 80',
            '6': 'maximum speed limit is 80 anymore',
            '7': 'maximum speed limit is 100',
            '8': 'maximum speed limit is 120',
            '9': 'overtaking is prohibited for all vehicles',
            '10': 'overtaking is prohibited for large vehicles',
            '11': 'priority at the upcoming intersection',
            '12': 'priority road starts',
            '13': 'yield right of way',
            '14': 'stop and yield',
            '15': 'no entry for any type of vehicle',
            '16': 'no entry for vehicles with a maximum authorized mass of more than 3.5 tonnes',
            '17': 'do not enter',
            '18': 'general danger or warning sign',
            '19': 'approaching single curve in the left direction',
            '20': 'approaching single curve in the right direction',
            '21': 'approaching double curve, first to the left',
            '22': 'rough road ahead',
            '23': 'danger of skidding or slipping',
            '24': 'the road narrows from the right side',
            '25': 'work in progress, be aware of workers on the road',
            '26': 'traffic signal ahead',
            '27': 'pedestrians may cross the road - installation on the right side of the road',
            '28': 'pay attention to children - installation on the right side of the road',
            '29': 'beware of cyclists',
            '30': 'beware of icy road ahead',
            '31': 'beware of wild animals',
            '32': 'end of all previously set passing and speed restrictions',
            '33': 'traffic must turn right',
            '34': 'traffic must turn left',
            '35': 'no turns permitted, mandatory traffic direction is ahead',
            '36': 'mandatorily, drive ahead or to the right',
            '37': 'mandatorily, drive ahead or to the left',
            '38': 'drive from the right of the obstacle',
            '39': 'drive from the left of the obstacle',
            '40': 'approaching entry to a roundabout',
            '41': 'end of overtaking restrictions for vehicles with a maximum authorized mass of less than 3.5 tonnes',
            '42': 'end of overtaking restrictions for all vehicles',
        }

    @route('/')
    def index(self):
        """
        renders the html page (the ui)
        :return: Void
        """
        return render_template_string(
            '''<!DOCTYPE html>

            <html>

            <head>
                <title>Traffic Sign Classifier</title>
            </head>

            <body>

            <img id="output" width="200" />
            <form id="my_form" method="POST" action="/predict" enctype="multipart/form-data">
                <input type="file" name="my_file" onchange="loadFile(event)" />
                <input type="submit" id="my_button" value="Predict" />
                <input type="submit" id="my_correct_button" value="Save" />
            </form>

            <div id="result"></div>
            <div id="notice"></div>

            <script>

            var loadFile = function(event) {
                var image = document.getElementById('output');
                image.src = URL.createObjectURL(event.target.files[0]);
                var res = document.getElementById('result');
                res.innerHTML = '';
                var notice = document.getElementById('notice');
                notice.innerHTML = '';
                
            };

            var my_form = document.getElementById("my_form");
            var my_file = document.getElementById("my_file");
            var my_button = document.getElementById("my_button");
            var my_button_correct = document.getElementById("my_correct_button");
            var result = document.getElementById("result");
            var notice = document.getElementById("notice");


            my_button.onclick = function(event){

                var formData = new FormData(my_form);
                formData.append('my_file', my_file);

                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/predict', true);

                xhr.addEventListener('load', function(e) {
                    result.innerHTML = xhr.response;
                    notice.innerHTML = 'If the prediction is correct, please hit Save'
                    
                });

                xhr.send(formData);

                event.preventDefault();
            };

            my_button_correct.onclick = function(event){

                var formData = new FormData(my_form);
                formData.append('my_file', my_file);

                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/save', true);

                xhr.addEventListener('load', function(e) {
                    result.innerHTML =xhr.response;
                    notice.innerHTML = ''

                });

                xhr.send(formData);

                event.preventDefault();
            };
            </script>

            </body>

            </html>'''
        )

    @route('/save', methods=['POST'])
    def save(self):
        """
        saves the image in the application/images/*classID* folder
        :return: String
        """
        self.model = load_model('E:/UBB CS/LICENTA/Model/Trained Model/Trained')
        temp_file = request.files.get('my_file')
        try:
            initialImg = imread(temp_file)
        except:
            return "please upload an image"
        img = cv2.resize(initialImg, (30, 30), interpolation=cv2.INTER_LINEAR)
        prediction = self.model.predict(img.reshape(-1, 30, 30, 3))
        classID = str(np.argmax(prediction))

        folderName = os.path.join('images', classID)
        try:
            list = os.listdir(folderName)
        except:
            os.makedirs(folderName)
            list = []
        list = os.listdir(folderName)
        imageNo = len(list)
        with open("images/imageCounter.txt", 'r') as file:
            content = str(file.read())
            name = content.strip()
            save_name = name.strip('\n')
        open('images/imageCounter.txt', 'w').close()
        with open("images/imageCounter.txt", 'w') as file:
            count = int(save_name) + 1
            count = str(count).strip()
            file.write(count)
        save_name = save_name + '.jpg'
        full_path = os.path.join(folderName, save_name)
        cv2.imwrite(full_path, cv2.cvtColor(initialImg, cv2.COLOR_RGB2BGR))

        list = os.listdir(folderName)
        imageNo2 = len(list)
        assert (imageNo + 1 == imageNo2) # check if the image has been added
        return "Thank you for improving the dataset!"

    @route('/predict', methods=['POST'])
    def predict(self):
        """
        gets a prediction for the user uploaded image from the trained model
        :return: String
        """
        temp_file = request.files.get('my_file')

        self.model = load_model('E:/UBB CS/LICENTA/Model/Trained Model/Trained')

        try:
            img = imread(temp_file)
        except:
            return "please upload an image"
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)
        prediction = self.model.predict(img.reshape(-1, 30, 30, 3))
        result = self.classes[str(np.argmax(prediction))]

        """
        testX = np.load('testX.npy')
        testY = np.load('testY.npy')
        roundedLabels = np.argmax(testY, axis=1)
        y_pred1 = self.model.predict(testX)
        y_pred = np.argmax(y_pred1, axis=1)
        print(precision_score(roundedLabels, y_pred, average="macro"))
        print(recall_score(roundedLabels, y_pred, average="macro"))
        print(f1_score(roundedLabels, y_pred, average="macro"))
        """


        return result


