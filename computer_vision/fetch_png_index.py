from PIL import Image
from PIL import ImagePalette
import numpy as np

image1 = "prediction_0.png"

img1 = Image.open(image1)

width, height = img1.size

print(width)
print(height)

colors1 = img1.getcolors()
pallette1 = img1.getpalette()

print(colors1)
print(pallette1)
