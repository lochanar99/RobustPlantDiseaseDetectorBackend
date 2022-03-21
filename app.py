import json
from flask import Flask, request, render_template, json
#from flask_pymongo import PyMongo
from PIL import Image
from h5py._hl import datatype
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import base64

model = load_model('train_data.hdf5')
model.load_weights('weights.hdf5')


# This function will return the label for the given image
def prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    pred = model.predict(img)
    y = np.argmax(pred)
    classname = pd.read_csv('classes.csv', encoding='latin1')
    result = classname.iloc[y][1:3]
    return result.to_json()
    #return "Predicted Disease is " + str(diseaseName) + " and Recommended Pesticide is " + str(pesticidename)



# initializing the flask and mongodb instance
app = Flask(__name__)


# app.config["MONGO_URI"] = "mongodb://localhost:27017/leafdiseasedetection"
# mongo = PyMongo(app)

# Currenlty I am uploading image thorough form request.
@app.route('/')
def index():
    return render_template('index.html')


# This route shall accept the file from your react native app to be uploaded to the server and it wll add that image to the database and
# the uploads folder where all the images will be stored
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        data = request.form['file']
        # wriet the data to a file

        with open('encoded.txt', 'w') as f:
            f.write(data)
        file = open('encoded.txt', 'rb')
        byte = file.read()
        file.close()

        decodeit = open('uploaded_image.jpg', 'wb')
        decodeit.write(base64.b64decode((byte)))
        decodeit.close()
        imageclass = prediction('uploaded_image.jpg')
        # mongo.db.image.insert({'db_image_name': data, 'image_class': imageclass})
        data = json.loads(imageclass)
        return "Predicted Disease is " + data['classes'] + " and Recommended Pesticide is " + data['pesticide']
    else:
        return "Error"


if __name__ == "__main__":
    app.run()
