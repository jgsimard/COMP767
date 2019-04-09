import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import gluoncv
import torch.nn as nn
import torchvision.models as models

import utils


def get_patches(voc_dataset, img_index, detected_class):
    image, label = voc_dataset[img_index]
    filtered_label = utils.filter_labels(label)
    bbs = utils.get_labels_bb(filtered_label)
    patches = [image[bb.y1:bb.y2, bb.x1:bb.x2].asnumpy() for bb in bbs]
    return patches, bbs

def get_hard_negatives(voc_dataset, img_index):
    image, label = voc_dataset[img_index]
    threshold = 0.3
    # we choose to do 1 positive : 3 negative
    nb_negative_for_each_positive = 3
    positives, positives_bb = get_patches(voc_dataset, img_index)

    negatives = []
    negatives_bbs = []
    i = 0
    while i < nb_negative_for_each_positive * len(positives):
        randoms = (np.random.rand(4) * min(image.shape[0], image.shape[1])).astype(int)
        x1 = int(min(randoms[0], randoms[1]))
        x2 = int(max(randoms[0], randoms[1]))
        y1 = int(min(randoms[2], randoms[3]))
        y2 = int(max(randoms[2], randoms[3]))
        bb = utils.BoundingBox(x1, y1, x2, y2)
        valid = True
        for positive_bb in positives_bb:
            iou = utils.intersection_over_union(bb, positive_bb)
            if iou > threshold:
                valid = False
                break
        if valid:
            negatives.append(image[bb.y1:bb.y2, bb.x1:bb.x2].asnumpy())
            negatives_bbs.append(bb)
            i += 1
    return negatives, negatives_bbs

def get_positives_and_negatives(root, voc_dataset, detected_class):
    indexes = utils.get_indexes_class(root, voc_dataset, detected_class)
    positives = []
    hard_negatives = []
    for index in indexes:
        patches, bbs = get_patches(voc_dataset, index)
        negatives, n_bbs = get_hard_negatives(voc_dataset, index)

        positives.extend([utils.resize_img(patch) for patch in patches])
        hard_negatives.extend([utils.resize_img(patch) for patch in negatives])
    return positives, hard_negatives

def get_class_dataset(root, detected_class):
    voc_dataset = gluoncv.data.VOCDetection(root=root, splits=[(2007, "trainval")])
    positives, hard_negatives = get_positives_and_negatives(root, voc_dataset, detected_class)
    x = torch.cat(torch.tensor(positives), torch.tensor(hard_negatives))
    y = torch.cat(torch.ones(len(positives)), torch.zeros(len(hard_negatives)))
    perm = torch.randperm(len(y))
    x = x[perm]
    y = y[perm]
    return x, y


class Evaluator:
    def __init__(self, nb_classes, root, make_from_scratch = False):
        self.root = root
        self.nb_classes = nb_classes

        self.base_cnn = models.vgg16(pretrained=True)
        self.base_cnn.classifier = nn.Sequential(*list(self.base_cnn.classifier.children())[:-2])
        for param in self.base_cnn.parameters():
            param.requires_grad = False
        self.base_cnn.to(self.device)

        self.output_layers = [nn.Linear(4096, 2).to(utils.get_device()) for _ in range(nb_classes)]
        self.relu = nn.ReLU()

        self.folder_name = "evaluator_layers"
        if make_from_scratch:
            os.mkdir("evaluator_layers")


    def forward(self, x, class_index):
        return self.relu(self.output_layers[class_index](self.base_cnn(x)))


    def train_class(self, class_index):
        x, y = get_class_dataset(self.root, class_index)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.output_layers[class_index])
        batch_size = 32
        n_batch_by_epoch = len(y)//batch_size
        for epoch in range(10):
            for i in range(n_batch_by_epoch):
                batch_x = x[i * batch_size:(i + 1) * batch_size]
                batch_y = y[i * batch_size:(i + 1) * batch_size]

                batch_y_pred = self.forward(batch_x, class_index)

                loss = criterion(batch_y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()




