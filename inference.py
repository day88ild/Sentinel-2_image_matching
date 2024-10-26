import numpy as np
import cv2
import matplotlib.pyplot as plt

import random
import os

import torch
import torchvision
from torchvision import transforms, models

import sklearn
from sklearn.metrics.pairwise import cosine_similarity


GREETING_TEXT = """Welcome to the Image Matching and Keypoint Visualization Program!

This tool allows you to compare two images and determine if they match by detecting and visualizing keypoints. Simply provide the paths to two images, and the program will output a similarity assessment along with a visualization of the keypoints and matches between the images.
Here are fast poss choices:

'images_to_choose/T36UXA_20180805T083559_TCI.jpg',
 'images_to_choose/T36UYA_20190601T083609_TCI.jpg',
 'images_to_choose/T36UXA_20190427T083601_TCI.jpg',
 'images_to_choose/T36UYA_20190621T083609_TCI.jpg',
 'images_to_choose/T36UXA_20180919T083621_TCI.jpg',
 'images_to_choose/T36UYA_20190412T083609_TCI.jpg',
 'images_to_choose/T36UXA_20180731T083601_TCI.jpg',
 'images_to_choose/T36UXA_20180904T083549_TCI.jpg',
 'images_to_choose/T36UYA_20190402T083559_TCI.jpg',
 'images_to_choose/T36UXA_20180825T083549_TCI.jpg',
 'images_to_choose/T36UYA_20190721T083609_TCI.jpg',
 'images_to_choose/T36UYA_20190422T083609_TCI.jpg',
 'images_to_choose/T36UXA_20180726T084009_TCI.jpg',
 'images_to_choose/T36UYA_20160502T083602_TCI.jpg',
 'images_to_choose/T36UXA_20180830T083601_TCI.jpg',
 'images_to_choose/T36UXA_20180815T084009_TCI.jpg',
 'images_to_choose/T36UYA_20190810T083609_TCI.jpg',
 'images_to_choose/T36UYA_20190701T083609_TCI.jpg',
 'images_to_choose/T36UYA_20160330T082542_TCI.jpg',
 'images_to_choose/T36UXA_20190606T083601_TCI.jpg',
 'images_to_choose/T36UYA_20190611T083609_TCI.jpg',
 'images_to_choose/T36UXA_20180820T083601_TCI.jpg',
 'images_to_choose/T36UXA_20180810T083601_TCI.jpg',
 'images_to_choose/T36UYA_20190517T083601_TCI.jpg',
 'images_to_choose/T36UYA_20160212T084052_TCI.jpg'
"""


def draw_keypoints(img1_path, img2_path):
    sift = cv2.SIFT_create()
    
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 20 matches
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    plt.figure(figsize=(15, 20))
    plt.imshow(result, cmap='gray')
    plt.axis("off")
    plt.show()


def image_similarity(img1_path, img2_path): # function that outputs similarity score between two images (0, 1)
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load pre-trained ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=True)
    
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity() # Change the last clf layer to Identity to get rid of it and get images features

    
    model.eval()  # Set to evaluation mode
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Transform images
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
       
    # Perform inference
    with torch.inference_mode():
        img1_output = model(img1).numpy().reshape(1, -1)
        img2_output = model(img2).numpy().reshape(1, -1)
    
    similarity_score = cosine_similarity(img1_output, img2_output) # compute similarity
    
    return similarity_score
    

def inference(img1_path, img2_path, threshold=0.45): # the inference function
    image_sim_score = image_similarity(img1_path, img2_path)[0, 0]
    
    if image_sim_score > threshold:
        print("The images are maching!!!")
    else:
        print("The images are not matching(")
    
    draw_keypoints(img1_path, img2_path)


if __name__ == "__main__":
    print(GREETING_TEXT)
    
    img1 = input("Enter path to the first image:")
    img2 = input("Enter path to the second image:")

    inference(img1, img2)
