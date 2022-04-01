from glob import glob

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import layers


from IPython.display import clear_output
from PIL import Image
from PIL import ImagePalette
from keras.backend import manual_variable_initialization 
import cv2

# For more information about autotune:
# https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

# important for reproducibility
# this allows to generate the same random numbers
SEED = 42
LEARNING_RATE = 0.00005

#LEARNING_RATE = 5e-3

# you can change this path to reflect your own settings if necessary
training_data = "training/images/"
training_annotations = "training/annotations/"
val_data = "validation/images/"
val_an = "validation/annotations/"

#Clear the training image directories
progression_graphs = glob('prediction_progression/*')
for f in progression_graphs:
    os.remove(f)

predictions = glob('predictions/*')
for f in predictions:
    os.remove(f)

frames_overlayed = glob('frames_overlayed/*')
for f in frames_overlayed:
    os.remove(f)

frames_masks = glob('frames_masks/*')
for f in frames_masks:
    os.remove(f)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# Image size that we are going to use
IMG_SIZE = 256
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 8
# Neural Network Kernel size
KERNEL_SIZE = 3

TRAINSET_SIZE = len(glob(training_data + "*.jpg"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(val_data + "*.jpg"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")



class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(epoch)
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        

def parse_image(img_path: str) -> dict:
   
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

   
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")

    mask = tf.io.read_file(mask_path)
   
    mask = tf.image.decode_png(mask, channels=1)
    
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
  

    return {'image': image, 'segmentation_mask': mask}


def parse_image2(img_path):
    
    image_new = tf.io.read_file(img_path)
    image_new = tf.image.decode_jpeg(image_new, channels=3)
    image_new = tf.image.resize(image_new, (128, 128), antialias=True)
    image_new = tf.cast(image_new, tf.float32) / 255.0
    
    image_new = image_new[tf.newaxis, ...]
    
    return image_new

def upscale_prediction(orig_path, prediction):
    
    image_orig = tf.io.read_file(orig_path)
    image_orig = tf.image.decode_jpeg(image_orig, channels=3)

    h = image_orig.shape[0]
    w = image_orig.shape[1]

    new_mask = tf.image.resize(prediction, (h, w))

    return new_mask


train_dataset = tf.data.Dataset.list_files(training_data + "*.jpg", seed=SEED)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(val_data + "*.jpg", seed=SEED)
val_dataset =val_dataset.map(parse_image)


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
   
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
  
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

BATCH_SIZE = 32

BUFFER_SIZE = 1000

dataset = {"train": train_dataset, "val": val_dataset}

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Validation Dataset --#
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

print(dataset['train'])
print(dataset['val'])


def display_sample(epoch, display_list):
   
    plt.clf()
    plt.figure(figsize=(18, 18))
   
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    
    plt.savefig('prediction_progression/' + str(epoch) + '_sample_prediction.png')
    plt.close()

input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
initializer = 'he_normal'

# -- UNET model (from scratch) -- #

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


# We use an even class distribution for this problem
def add_sample_weights(image, label):

  class_weights = tf.constant([0.06, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
  class_weights = class_weights/tf.reduce_sum(class_weights)

  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    
    pred_mask = tf.argmax(pred_mask, axis=-1)
    
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
    
def show_predictions(epoch, num=1):
    
    for image, mask in dataset['train'].take(1):
        sample_image, sample_mask = image, mask

    image = sample_image[0][tf.newaxis, ...]
    
    #test_image_url = 'test_image.jpg'
    #test_image = parse_image2(test_image_url)

    inference = model.predict(image)
    pred_mask = create_mask(inference)
    
    np.savetxt("prediction_mask_train.csv", tf.squeeze(pred_mask), delimiter=",")
    tf.keras.preprocessing.image.save_img("predictions/prediction_" + str(epoch) + ".png", tf.squeeze(pred_mask, [0]), data_format=None, file_format=None, scale=True)
    display_sample(epoch, [sample_image[0], sample_mask[0], pred_mask[0]])

def overlay_transparent(bg_img, img_to_overlay_t):
   
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)

    added_image = cv2.addWeighted(bg_img,1,img_to_overlay_t,0.4,0)

    return added_image

def make_transparent(file_name):

    src = cv2.imread(file_name, 1)
    
    
    src[src != 255] = 0

   
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,60,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)

    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba)

    return dst

def convert2RGB(prediction):

    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.expand_dims(prediction, axis=-1)

    prediction = tf.squeeze(prediction, [0])

    prediction = np.where(prediction == 0, [0, 0, 0], prediction)
    prediction = np.where(prediction == 1, [245, 245, 66], prediction)
    

    prediction = np.where(prediction == 2, [66, 135, 245], prediction)
    prediction = np.where(prediction == 3, [66, 135, 245], prediction)
    prediction = np.where(prediction == 4, [66, 135, 245], prediction)

   
    return prediction

def create_video_images():
    frames_dir = glob('frames_all/*')

    for filename in frames_dir:
        print(filename)
        file_path = os.path.basename(filename)
        file_name_only = os.path.splitext(file_path)[0]
        test_image_url = filename
        test_image = parse_image2(test_image_url)
        inference = model.predict(test_image)
        RGB_Mask = convert2RGB(inference)

        pred_mask = upscale_prediction(filename, RGB_Mask)
        tf.keras.preprocessing.image.save_img("frames_masks/" + file_name_only + ".png", pred_mask, data_format=None, file_format=None, scale=True)

        maskimg = cv2.imread("frames_masks/" + file_name_only + ".png")
        originalimg = cv2.imread("frames_all/" + file_name_only + ".jpg")

        trans_mask = make_transparent("frames_masks/" + file_name_only + ".png")
        overlay = overlay_transparent(originalimg, trans_mask)

        cv2.imwrite("frames_overlayed/" + file_name_only + ".png", overlay)


#show_predictions()

EPOCHS = 80
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [
    # to show samples after each epoch
    DisplayCallback(),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('model_unet_check.h5', verbose=1, save_best_only=True, save_weights_only=True)
    
]

#model = tf.keras.Model(inputs = inputs, outputs = output)
model = DeeplabV3Plus(image_size=IMG_SIZE, num_classes=N_CLASSES)
print(model.summary())

# Functional
optimizer = Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss = loss,
                  metrics=['accuracy'])



model_history = model.fit(dataset['train'].map(add_sample_weights), 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=dataset['val'].map(add_sample_weights),
                          callbacks=[callbacks])


    
model.save('final_' + str(EPOCHS) + '.h5')

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)
plt.clf()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 0.3])
plt.legend()
plt.savefig('TrainingLoss.png')


#create_video_images()


