from PIL import Image
from PIL import ImagePalette
import numpy as np
import io
import os
import cv2

masks_directory = "masks/"
training_direction = "training/annotations/"
validation_direction = "validation/annotations/"

color_array = []

for file in os.listdir(masks_directory):
    #filename = os.fsdecode(file)
    
    img = Image.open(masks_directory + file)
    width, height = img.size

    img = img.convert("RGBA")
    colors = img.getcolors()
    color = colors[0][1]

    if not color in color_array:
        print("Color is in array already.")
        color_array.append(color)

print(len(color_array))
print(color_array)

for file in os.listdir(training_direction):
    img = Image.open(training_direction + file)
    width, height = img.size

    colors = img.getcolors()

    #print(colors)
    if len(colors) < 2:
        print("Color missing from file: " + file)


for file in os.listdir(validation_direction):
    img = Image.open(validation_direction + file)
    width, height = img.size

    colors = img.getcolors()

    #print(colors)
    if len(colors) < 2:
        print("Color missing from file: " + file)
