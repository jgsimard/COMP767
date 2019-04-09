import cv2
import gluoncv
import gym
import numpy as np
from gym import spaces
import copy
from matplotlib import pyplot as plt
import torch

class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def area(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            return 0
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __repr__(self):
        return f"(x1,y1)=({self.x1}, {self.y1}), (x2,y2)=({self.x2}, {self.y2})"


def intersection_over_union(bb1, bb2):
    intersection_bb = BoundingBox(x1=max(bb1.x1, bb2.x1),
                                  y1=max(bb1.y1, bb2.y1),
                                  x2=min(bb1.x2, bb2.x2),
                                  y2=min(bb1.y2, bb2.y2))
    iou = intersection_bb.area() / (bb1.area() + bb2.area() - intersection_bb.area())
    if iou > 1:
        print("FAULTY IOU", iou)
        print("bb1", bb1)
        print("bb2", bb2)
        print("intersection_bb", intersection_bb)
    return iou


def get_indexes_class(root, voc_dataset, detected_class, year=2007, set_name="trainval"):
    fname = f"{root}/VOC{year}/ImageSets/Main/{voc_dataset.classes[detected_class]}_{set_name}.txt"
    with open(fname) as f:
        content = f.readlines()

    content_index = [int(x.strip().split()[0]) for x in content]
    content_detected_class = [int(x.strip().split()[0]) for x in content if int(x.strip().split()[1]) == 1]
    content_detected_class_filtered = []
    i = 0
    for class_img_index in content_detected_class:
        while True:
            if content_index[i] == class_img_index and i < len(content_index):
                content_detected_class_filtered.append(i)
                break
            i += 1
    return content_detected_class_filtered


def filter_labels(labels, detected_class):
    return labels[labels[:, 4] == detected_class]


def get_labels_bb(labels):
    bbs = []
    for i in range(labels.shape[0]):
        bbs.append(BoundingBox(int(labels[i, 0]), int(labels[i, 1]), int(labels[i, 2]), int(labels[i, 3])))
    return bbs

def resize_img(img, output_image_size=224):
    return cv2.resize(img, (output_image_size, output_image_size))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
