from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import base64

model = load_model('train_data.hdf5')
model.load_weights('weights.hdf5')

def prediction(img_path):
    # PIL image loading
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    pred = model.predict(img)
    y = np.argmax(pred)
    print(y)
    classes = pd.read_csv('classes.csv', encoding='latin1')
    prediction = classes.iloc[1][1:3]
    return prediction.to_json()


# file = open('encoded.txt', 'rb')
# byte = file.read()
# file.close()

# decodeit = open('uploaded_image.jpg', 'wb')
# decodeit.write(base64.b64decode((byte)))
# decodeit.close()

    

print(prediction('images/blackrot.jpg'))