import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


class ImageLabelParser:

    @staticmethod
    def faces(labels_folder, images_folder, label_name):
        # receive image label (xml file)
        # return mask_faces[] and no_mask_faces[]

        mask_faces, no_mask_faces = [], []

        annotation = minidom.parse(labels_folder + label_name)

        image_name = annotation.getElementsByTagName('filename')[0].firstChild.data
        image = cv2.imread(images_folder + image_name)

        objects = annotation.getElementsByTagName('object')
        for obj in objects:
            bndbox = obj.getElementsByTagName('bndbox')[0]
            xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
            face = image[ymin:ymax + 1, xmin:xmax + 1]

            label = obj.getElementsByTagName('name')[0].firstChild.data
            if label == 'mask':
                mask_faces.append(face)
            elif label == 'none':
                no_mask_faces.append(face)

        return mask_faces, no_mask_faces
