import cv2
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import face_dataset
from mask_classifier import train
from mask_classifier import MaskClassifier


IMAGES_PASS = 'keggle_dataset/medical-masks-dataset/images/'


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

#train()

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_classifier = MaskClassifier()
i = 0
for image in os.listdir(IMAGES_PASS):
    image = cv2.imread(IMAGES_PASS + image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(img_gray, 1.1, 19)
    for x, y, w, h in faces:
        face = image[y:y + h, x:x + w]
        color = (0, 255, 0) if mask_classifier.is_in_mask(face) else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    i += 1
    cv2.imshow('', image)
    cv2.imwrite('figures/' + str(i) + '.jpg', image)
    cv2.waitKey(0)


#  labels_folder = 'keggle_dataset/medical-masks-dataset/labels/'
#  images_folder = 'keggle_dataset/medical-masks-dataset/images/'




