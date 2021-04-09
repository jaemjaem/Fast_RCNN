from torch.utils.data import DataLoader
from dataset import Fast_RCNN_Dataset
from model import Fast_RCNN
from utils.iou import intersection_over_union

import torch
import numpy as np
import torchvision.transforms as transforms
import selective_search

image_path = "/mnt/VOC2012/image/"
label_path = "/mnt/VOC2012/anno/"

def multi_task_loss(target, classifier_vector, bbox_vector):
    

    return None

def train():
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Fast_RCNN_Dataset(image_path, label_path, transform)
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2)

    net = Fast_RCNN().cuda()

    for i, batch in enumerate(train_loader):
        img, positive_sampling, negative_sampling = batch
        
        img = img.cuda()
        sampling = torch.cat([positive_sampling, negative_sampling], dim=1)

        classifier_vector, bbox_vector = net(img, sampling)
        print(classifier_vector.shape)
        print(bbox_vector.shape)
        
if __name__ == "__main__":
    train()