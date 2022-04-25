# Predicting Style with Computer Vision #

As a supplement to the clothing classifier, a computer vision algorithm is constructed to analyze the size, shape and color of each garmet in the dataset to better interpret style. Instance segmentation is used for the pixel-wise classification for this project. This ensures the exact geometry and dimensions of a given garmet are captured and those features are used to provide better discrimination and classification of our 'style' feature.

## Algorithm, Machine Learning Framework and Data Selection ##

We use a series of 971 images split into a training (80%) validation (10%) and test set (10%) in order to both train our classifier and validate its performance. The images themselves are scraped from the ecommerce website. Ground truth pixel masks are generated manually for all 971 images in the dataset, where 97 of those are used as a hold-out set to prove the accuracy of the model.

A deep neural network system, specifically a convultional neural network (CNN) is used for building the feature maps and computing the weights and biases used to infer our pixel-wise classification and eventual categorization of the images:

Example of a scrapped image:

![Scheme](original_images/4011_alternative.jpg)


Example of a generated mask:

![Scheme](masks/4011_alternative.png)
