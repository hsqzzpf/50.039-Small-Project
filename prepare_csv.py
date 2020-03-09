import os
import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch


directory = "../code/data/VOC2012"
def get_label_dict():
    return {
            'aeroplane' :    0,
            'bicycle' :      1,
            'bird' :         2,
            'boat' :         3,
            'bottle' :       4,
            'bus' :          5,
            'car' :          6,
            'cat' :          7,
            'chair' :        8,
            'cow' :          9,
            'diningtable' :  10,
            'dog' :          11,
            'horse' :        12,
            'motorbike' :    13,
            'person' :       14,
            'pottedplant' :  15,
            'sheep' :        16,
            'sofa' :         17,
            'train' :        18,
            'tvmonitor' :    19
        }


def _get_images_list(split):
    image_paths = []
    image_path_file = os.path.join(directory, 'ImageSets/Main', split + '.txt')
    with open(image_path_file) as f:
        for image_path in f.readlines():
            candidate_path = image_path.split(' ')[0].strip('\n')
            image_paths.append(candidate_path)
    return image_paths

def _get_xml_file_path(image_name):
    xml_name = image_name + '.xml'
    xml_path = os.path.join(directory, 'Annotations', xml_name)
    return xml_path

verbose = False
def _load_all_image_paths_labels(split):
    label_count = 0
    all_image_paths_labels = []
    images_list = _get_images_list(split)
    xml_path_list = [_get_xml_file_path(image_path)
                        for image_path in images_list]
    for image_path, xml_path in zip(images_list, xml_path_list):
        image_path = os.path.join(directory, 'JPEGImages', image_path + '.jpg')
        assert(image_path not in all_image_paths_labels)
        # if self.multi_instance:
        if True:
            labels = _get_labels_from_xml(xml_path)
        else:
            labels = list(np.unique(_get_labels_from_xml(xml_path)))
        label_count += len(labels)
        if verbose:
            print("Loading labels of size {} for {}...".format(
                 len(labels), image_path))
        image_path_labels = {'image_path': image_path,
                            'labels': labels}
        all_image_paths_labels.append(image_path_labels)

    print("SET: {} | TOTAL IMAGES: {}".format(split, len(all_image_paths_labels)))
    print("SET: {} | TOTAL LABELS: {}".format(split, label_count))
    return label_count, all_image_paths_labels


def _get_labels_from_xml(xml_path):
    labels = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root.iter('object'):
        labels.append(child.find('name').text)
    return labels


def convert_oh(ls, length=20):

    output = [0] * length

    for idx in ls:
        output[idx] = 1
    return output


import csv
import pandas as pd

def make_csv(addr):

    if os.path.exists(addr):
        return

    out = open(addr, 'a')
    wr = csv.writer(out, dialect='excel')

    _, data = _load_all_image_paths_labels("train")
    
    for i in range(len(data)):
        labels = []
        for label in data[i]["labels"]:
            labels.append(get_label_dict()[label])
        wr.writerow([data[i]["image_path"], convert_oh(labels)])
    


if __name__ == "__main__":
    make_csv("./dd.csv")
    print("finish writing")

    df = pd.read_csv("./dd.csv")
    print(df)


