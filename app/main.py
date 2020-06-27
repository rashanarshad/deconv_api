from deepdream import visualize_all_layers, deprocess_image
from keras.applications import vgg16
import keras.backend as K
from keras.preprocessing import image
import keras.backend.tensorflow_backend as tb

from keras.applications.vgg16 import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form
import base64
import cv2
import numpy as np
from urllib.parse import quote
from base64 import b64encode

K.set_image_data_format('channels_last')
model = vgg16.VGG16(weights='imagenet', include_top=True)

app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post('/')
async def return_deconv(file: str = Form(...), layer: str = Form(...)):

    # convert string of image data to uint8
    img64 = file
    img = readb64(img64)

    #resize image for model
    img = cv2.resize(img, (224, 224))
    tb._SYMBOLIC_SCOPE.value = True

    #set chosen layer name for viewing deconv
    layer_name = layer

    #preprocess before visualizing
    in_array = np.expand_dims(image.img_to_array(img), axis=0)
    img_array = preprocess_input(in_array)

    #visualize layers
    deconv = visualize_all_layers(model, img_array, layer_name=layer_name, visualize_mode='all')

    #stitch together top 4 filters into one
    top_img = np.concatenate((deconv[layer_name][0], deconv[layer_name][1]), axis=1)
    bottom_img = np.concatenate((deconv[layer_name][2], deconv[layer_name][3]), axis=1)
    concat_img = np.concatenate((top_img, bottom_img), axis=0)

    #deprocess image and convert back to base64 representation as string
    total_concat = deprocess_image(concat_img)
    retval, buffer = cv2.imencode('.jpg', total_concat)
    data = b64encode(buffer)
    data = data.decode('ascii')
    data_url = 'data:image/webp;base64,{}'.format(quote(data))

    return data_url
