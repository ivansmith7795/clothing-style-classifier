from glob import glob

import pandas as pd
import numpy as np
import urllib.request as urllib2
from bs4 import BeautifulSoup

import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import layers

from IPython.display import clear_output
from keras_radam import RAdam
from PIL import Image
from PIL import ImagePalette


# Image size that we are going to use
IMG_SIZE = 256
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene has 5 classes
N_CLASSES = 8
# Neural Network Kernel size
KERNEL_SIZE = 3

input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

    
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    ):

    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

def parse_image(img_path):
    
    image_new = tf.io.read_file(img_path)
    image_new = tf.image.decode_jpeg(image_new, channels=3)
    image_new = tf.image.resize(image_new, (256, 256))
    image_new = tf.cast(image_new, tf.float32) / 255.0
    
    image_new = image_new[tf.newaxis, ...]
    
    return image_new

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
   
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def get_class_prediction(inference: tf.Tensor):

    classes = ['unknown', 'classic', 'sporty', 'girly', 'edgy', 'casual chic', 'trendy', 'alternative']
    class_map = dict(map(lambda t: (t[1], t[0]), enumerate(classes)))

    predicted_class = 0
    array_size = 0
    
    # Let's find the probable class
    for i in range(8):
        if i > 0:
            mask_index = inference[:, :, :, i]
            mask_size = tf.math.reduce_sum(mask_index)
            #tf.print(mask_size)
            if mask_size > array_size:
                array_size = mask_size
                predicted_class = i 

    if array_size < 5000:
        predicted_class = 0
    
    prediction = list(class_map.keys())[list(class_map.values()).index(predicted_class)]
    return prediction

def scrape_image(index, dress_page):
    #dress_page = dress_page.replace(" ","")

    parsed_image = []

    page = None
    print(dress_page)
    dress_page = dress_page.replace(" ","")

    try:
        page = urllib2.urlopen(dress_page)
        
    except urllib2.HTTPError as e:
        print(str(index) + ' could not find: ' + dress_page + '. Failed with error code: ' + str(e.code))
    
    # Parse the page if returned data
    if page is not None:

        soup = BeautifulSoup(page, 'html.parser')

        images = []
        for img in soup.findAll('img', attrs={'class': 'FxZV-M'}):
            if "packshot" not in img.get('src'): 
                continue
            else:
                images.append(img.get('src'))
                break

        if len(images) > 0:
            urllib2.urlretrieve(images[0], "scraped_labeled/" + str(index) + ".jpg")
            parsed_image = parse_image("scraped_labeled/" + str(index) + ".jpg")
    
    return parsed_image

#Load our model
model = DeeplabV3Plus(image_size=IMG_SIZE, num_classes=N_CLASSES)
model.load_weights('final_80.h5')
weights = model.get_weights()

print(model.summary())


#Import into pandas
labeled_dataset = pd.read_csv('../datasets/dress-dataset-labeled-processed.csv')
predicted_classes = []

for index, row in labeled_dataset.iterrows():
    predicted_class = 'unknown'
    page_url = row['Link']
    index_id = row['IndexID']
    image = scrape_image(index_id, page_url)
    
    if len(image) > 0:
        inference = model.predict(image)
        predicted_class = get_class_prediction(inference)
        print(str(index_id) + ' ' + str(predicted_class))
    
    predicted_classes.append(predicted_class)

labeled_dataset.insert(0, "cv_predicted", predicted_classes)
labeled_dataset.to_csv('../datasets/dress-dataset-labeled-processed.csv', index=False)

#Save the dataframe 
#test_image_url = '1553_classic.jpg'
#test_image = parse_image(test_image_url)

#inference = model.predict(test_image)

#print(get_class_prediction(inference))

#pred_mask = create_mask(inference)

#pred_mask = tf.squeeze(pred_mask, [0])
#test_image = tf.squeeze(test_image, [0])

#tf.keras.preprocessing.image.save_img('test_image_mask.png', pred_mask, data_format=None, file_format=None, scale=True)
#tf.keras.preprocessing.image.save_img('test_image_original.jpg', test_image, data_format=None, file_format=None, scale=True)


