import comet_ml
import os

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils
import utils

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])
def get_dataset_dataloader(detected_class, batch_size):
    dataset = torchvision.datasets.ImageFolder(root=f"./dataset/{detected_class}", transform=image_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)
    return dataset, dataset_loader



def get_classifier():
    return nn.Sequential(nn.Linear(4096, 200),
                         nn.ReLU(inplace=True),
                         nn.Dropout(p=0.5),
                         nn.Linear(200, 1),
                         nn.Sigmoid()).to(utils.get_device())


class Evaluator:
    def __init__(self, nb_classes, root, make_from_scratch=False):
        self.experiment = utils.get_experiment()
        self.root = root
        self.nb_classes = nb_classes
        self.batch_size = 32
        self.nb_epochs = 30
        self.base_cnn = models.vgg16(pretrained=True)
        self.base_cnn.classifier = nn.Sequential(*list(self.base_cnn.classifier.children())[:-4])
        for param in self.base_cnn.parameters():
            param.requires_grad = False
        self.base_cnn.to(utils.get_device())

        self.classifiers = [get_classifier() for _ in range(nb_classes)]
        if not make_from_scratch:
            for i in range(nb_classes):
                utils.load_model(self.classifiers[i], os.path.join("evaluator", f"{i}_classifier.pt"))

    def forward(self, x, class_index):
        return self.classifiers[class_index](self.base_cnn(x))

    def train_class(self, class_index):
        print(f"train class {class_index}")
        dataset, dataset_loader = get_dataset_dataloader(class_index, self.batch_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.classifiers[class_index].parameters())
        for epoch in range(self.nb_epochs):
            for i, (x, y) in enumerate(dataset_loader):
                x, y = x.to(utils.get_device()), y.to(utils.get_device())
                optimizer.zero_grad()
                y_pred = self.forward(x, class_index)
                loss = criterion(y_pred.view(-1), y.float())
                loss.backward()
                optimizer.step()
                self.experiment.log_metric(f"Class {class_index} loss", loss.item())

        evaluator_dir = "evaluator"
        utils.make_dir_if_not_exist(evaluator_dir)
        model_path = os.path.join(evaluator_dir, f"{class_index}_classifier.pt")
        utils.save_model(self.classifiers[class_index], model_path)


if __name__ == "__main__":
    evalutator = Evaluator(20, root="./", make_from_scratch=True)
    for i in range(20):
        evalutator.train_class(i)