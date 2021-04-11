import os
import cv2
import torch
import selective_search
import xml.etree.ElementTree as ET

from utils.iou import intersection_over_union
from torch.utils.data.dataset import Dataset

class Fast_RCNN_Dataset(Dataset):
    def __init__(self, image_folder_path, label_folder_path, N, K, transform):
        super(Fast_RCNN_Dataset, self).__init__
        img_filename_list = sorted(os.listdir(image_folder_path))
        label_filename_list = sorted(os.listdir(label_folder_path))

        self.K = K
        positive_num = int(N * 0.25)
        negative_num = int(N * 0.75)

        self.transform = transform

        self.data = []
        self.positive_sampling = []
        self.negative_sampling = []

        self.target_mask = []
        self.target_class = []
        self.target_bbox = []

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

            target_bbox_mask = []
            cls_list = []
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

                if(obj.find('name').text == 'person'):
                    cls = [0, 1] # 첫번째 배경 , 두번째부터 해당클래스로다가..

                for idx in range(len(regions)):
                    iou = intersection_over_union(resize_box, regions[idx])
                    if iou > 0.5 and len(positive) < positive_num:
                        positive.append(regions[idx])
                        cls_list.append(cls)
                        for i in range(len(cls)):
                            obj_list.append(box)
                            target_bbox_mask.append([1, 1, 1, 1])
                    elif iou >= 0.1 and iou < 0.5 and len(negative) < negative_num:
                        negative.append(regions[idx])
                        cls_list.append(cls)
                        for i in range(len(cls)):
                            obj_list.append(box)
                            target_bbox_mask.append([0, 0, 0, 0])

            self.target_mask.append(target_bbox_mask)
            self.target_class.append(cls_list)
            self.target_bbox.append(obj_list)
            self.positive_sampling.append(positive)
            self.negative_sampling.append(negative)

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        image = self.data[idx]

        target_class = torch.LongTensor(self.target_class[idx])
        target_bbox = torch.LongTensor(self.target_bbox[idx])

        target_mask = torch.Tensor(self.target_mask[idx])
        target_mask = target_mask.view(-1, self.K + 1, target_mask.size(1))

        if self.transform:
            transform_image = self.transform(image)
        
        positive_sampling = torch.Tensor(self.positive_sampling[idx])
        negative_sampling = torch.Tensor(self.negative_sampling[idx])

        return transform_image, target_class, target_bbox, target_mask, positive_sampling, negative_sampling  # 이미지, 타켓, postive, negative 순서