import os
import sys

import cv2
import torch
from comet_ml import Experiment
import matplotlib.pyplot as plt
import torchvision.utils



##################
# Logging
##################
def get_experiment():
    return Experiment(api_key="H5Zg5SDrkQeX0bL0sWyGSCdHl",
                      project_name="comp767_project",
                      workspace="jgsimard")


##################
# Folder stuff
##################

def make_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_directory_name_with_number(base_path):
    i = 0
    while os.path.exists(base_path + "_" + str(i)):
        i += 1
    return base_path + "_" + str(i)


def setup_run_folder(args):
    argsdict = args.__dict__
    argsdict['code_file'] = sys.argv[0]
    argsdict['device'] = get_device()

    runs_directory = "runs"
    run_name = f"detected_class={argsdict['detected_class']}"
    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory)
    base_path = os.path.join(runs_directory, run_name)
    experiment_path = get_directory_name_with_number(base_path)
    os.mkdir(experiment_path)

    print("Putting log in %s" % experiment_path)
    argsdict['save_dir'] = experiment_path
    with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
        for key in sorted(argsdict):
            f.write(key + '    ' + str(argsdict[key]) + '\n')

    experiment = get_experiment()
    experiment.log_parameters(argsdict)

    return experiment_path, experiment


##################
# BOUNDING BOX
##################

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


##################
# VOC PASCAL
##################
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


##################
# TORCH
##################

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location= get_device()))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    if len(inp.size()) == 4:
        inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    inp = (inp - inp.min()) * 1.0 / (inp.max() - inp.min())
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
