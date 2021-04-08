import os
import cv2
import numpy as np
import torch
import selective_search
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import selective_search

from utils.iou import intersection_over_union
from torch.utils.data.dataset import Dataset
from skimage import io, transform
from PIL import Image


class Fast_RCNN_Dataset(Dataset):
    def __init__(self, image_folder_path, label_folder_path, transform):
        super(Fast_RCNN_Dataset, self).__init__
        img_filename_list = sorted(os.listdir(image_folder_path))
        label_filename_list = sorted(os.listdir(label_folder_path))
        self.transform = transform

        self.data = []
        self.positive_sampling = []
        self.negative_sampling = []
        self.target = []

        for img_filename, label_filename in zip(img_filename_list, label_filename_list):
            img_path = image_folder_path + img_filename
            label_path = label_folder_path + label_filename
            tree = ET.parse(label_path)
            root = tree.getroot()
    
            image_size = root.find('size')
            image_width = int(image_size.find('width').text)
            image_height = int(image_size.find('height').text)

            img = cv2.imread(img_path)
            self.data.append(img)
            resized_img = cv2.resize(img, (224,224))
            regions = selective_search.selective_search(resized_img, mode='quality', random_sort=False)

            obj_list = []
            positive = []
            negative = []
            for obj in root.findall('object'):
                xmlbox = obj.find('bndbox')
                x1 = int(xmlbox.find('xmin').text) / image_width
                y1 = int(xmlbox.find('ymin').text) / image_height
                x2 = int(xmlbox.find('xmax').text) / image_width
                y2 = int(xmlbox.find('ymax').text) / image_height
                box = [x1, y1, x2, y2]
                resize_box = [int(224*x1), int(224*y1), int(224*x2), int(224*y2)]
                for region in regions:
                    iou = intersection_over_union(resize_box, region)
                    if iou > 0.5 and len(positive) < 16:
                        positive.append(region)
                    elif iou >= 0.1 and iou < 0.5 and len(negative) < 48:
                        negative.append(region)
             
            self.positive_sampling.append(positive)
            self.negative_sampling.append(negative)

        # self.sampling = []
        # for idx in range(len(self.positive_sampling)):
        #     self.sampling.append(self.positive_sampling[idx] + self.negative_sampling[idx])
        # print(len(self.sampling))
        
                    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        image = self.data[idx] 

        if self.transform:
            transform_image = self.transform(image)
        
        positive_sampling = torch.Tensor(self.positive_sampling[idx])
        negative_sampling = torch.Tensor(self.negative_sampling[idx])

        return transform_image, positive_sampling, negative_sampling # 이미지, 타켓, postive, negative 순서