import os
import cv2
import torch
import numpy as np
from image_label_parser import ImageLabelParser

MASK_FOLDER = 'mask_faces/'
NO_MASK_FOLDER = 'no_mask_faces/'
IMAGE_SIZE = (32, 32)


class Dataset:
    def __init__(self):
        # creates train_data, train_labels, test_data, test_labels

        self.MASK_AMOUNT = 750
        self.NO_MASK_AMOUNT = 750
        self.TRAIN_AMOUNT = int(0.75 * (self.MASK_AMOUNT + self.NO_MASK_AMOUNT))

        mask_faces = torch.tensor([image_prepare(cv2.imread(MASK_FOLDER + name))
                                   for name in os.listdir(MASK_FOLDER)[:self.MASK_AMOUNT]])
        mask_labels = torch.ones((len(mask_faces)))
        no_mask_faces = torch.tensor([image_prepare(cv2.imread(NO_MASK_FOLDER + name))
                                      for name in os.listdir(NO_MASK_FOLDER)[:self.NO_MASK_AMOUNT]])
        no_mask_labels = torch.zeros((len(no_mask_faces)))

        data = torch.cat([mask_faces, no_mask_faces], dim=0)
        labels = torch.cat([mask_labels, no_mask_labels], dim=0)

        order = np.random.permutation(len(data))
        data, labels = data[order], labels[order]

        self.train_data = data[:self.TRAIN_AMOUNT]
        self.train_labels = labels[:self.TRAIN_AMOUNT]
        self.test_data = data[self.TRAIN_AMOUNT:]
        self.test_labels = labels[self.TRAIN_AMOUNT:]


def image_prepare(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.array([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
    return image


def generate_faces(labels_folder, images_folder):
    # receive labels and images folders
    # write all mask_faces and no_mask_faces to the appropriate folders

    labels = os.listdir(labels_folder)
    i1, i2 = 0, 0
    for label_name in labels:
        mask_faces, no_mask_faces = ImageLabelParser.faces(labels_folder, images_folder, label_name)
        for face in mask_faces:
            i1 += 1
            try:
                cv2.imwrite(MASK_FOLDER + str(i1) + '.jpg', face)
            except Exception:
                print('er')
        for face in no_mask_faces:
            i2 += 1
            try:
                cv2.imwrite(NO_MASK_FOLDER + str(i2) + '.jpg', face)
            except Exception:
                print('er')
