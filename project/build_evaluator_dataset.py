import argparse
import os

import gluoncv
import matplotlib
import numpy as np
from tqdm import tqdm

import utils
from utils import BoundingBox, intersection_over_union, get_labels_bb, filter_labels, resize_img

parser = argparse.ArgumentParser(description='build dataset for region ranking')
parser.add_argument('--root', type=str,
                    default='./data/VOCtrainval_06-Nov-2007/VOCdevkit',
                    help='location of the data')
args = parser.parse_args()


def get_patches(voc_dataset, detected_class, img_index):
    image, label = voc_dataset[img_index]
    filtered_label = filter_labels(label, detected_class)
    bbs = get_labels_bb(filtered_label)
    patches = [image[bb.y1:bb.y2, bb.x1:bb.x2].asnumpy() for bb in bbs]
    return patches, bbs


def get_hard_negatives(voc_dataset, detected_class, img_index):
    image, label = voc_dataset[img_index]
    threshold = 0.3
    # we choose to do 1 positive : 3 negative
    nb_negative_for_each_positive = 3
    positives, positives_bb = get_patches(voc_dataset, detected_class, img_index)

    negatives = []
    negatives_bbs = []
    i = 0
    while i < nb_negative_for_each_positive * len(positives):
        randoms = (np.random.rand(4) * min(image.shape[0], image.shape[1])).astype(int)
        x1 = int(min(randoms[0], randoms[1]))
        x2 = int(max(randoms[0], randoms[1]))
        y1 = int(min(randoms[2], randoms[3]))
        y2 = int(max(randoms[2], randoms[3]))
        bb = BoundingBox(x1, y1, x2, y2)
        valid = True
        for positive_bb in positives_bb:
            iou = intersection_over_union(bb, positive_bb)
            if iou > threshold:
                valid = False
                break
        if valid:
            negatives.append(image[bb.y1:bb.y2, bb.x1:bb.x2].asnumpy())
            negatives_bbs.append(bb)
            i += 1
    return negatives, negatives_bbs


voc_dataset = gluoncv.data.VOCDetection(root=args.root, splits=[(2007, "trainval")])
nb_real_class = len(voc_dataset.classes)

dataset_directory = "dataset"
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

for detected_class in tqdm(range(len(voc_dataset.classes))):
    detected_class_dir = os.path.join(dataset_directory, str(detected_class))
    detected_class_real_dir = os.path.join(detected_class_dir, "real")
    detected_class_fake_dir = os.path.join(detected_class_dir, "fake")
    utils.make_dir_if_not_exist(detected_class_dir)
    utils.make_dir_if_not_exist(detected_class_real_dir)
    utils.make_dir_if_not_exist(detected_class_fake_dir)

    indexes = utils.get_indexes_class(args.root, voc_dataset, detected_class)

    positives = []
    hard_negatives = []
    for index in indexes:
        patches, bbs = get_patches(voc_dataset, detected_class, index)
        negatives, n_bbs = get_hard_negatives(voc_dataset, detected_class, index)

        positives.extend(patches)
        hard_negatives.extend(negatives)


    def write_imgs_set(path, img_set, img_counter):
        for img in img_set:
            img_name = os.path.join(path, f"{img_counter}.png")
            matplotlib.image.imsave(img_name, img)
            img_counter += 1
        return img_counter

    img_counter = 0
    img_counter = write_imgs_set(detected_class_fake_dir, hard_negatives, img_counter)
    write_imgs_set(detected_class_real_dir, positives, img_counter)

