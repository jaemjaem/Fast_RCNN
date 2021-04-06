from torch.utils.data import DataLoader
from dataset import Fast_RCNN_Dataset
from model import Fast_RCNN
from utils.iou import intersection_over_union

import numpy as np
import torchvision.transforms as transforms
import selective_search

image_path = "/mnt/VOC2012/image/"
label_path = "/mnt/VOC2012/anno/"

def train():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Fast_RCNN_Dataset(image_path, label_path, transform)
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2)

    net = Fast_RCNN().cuda()

    # for i, batch in enumerate(train_loader):
    #     inputs = batch
    #     np_img = inputs.numpy()
        
    #     inputs = inputs.cuda()

    #     batch_region = []
    #     for idx in range(len(np_img)):
    #         regions = selective_search.selective_search(np.transpose(np_img[idx]), mode='fast', random_sort=False)
    #         for region in regions:
    #             iou = intersection_over_union()
        
    #     output = net(inputs, batch_region)
    #     print(output.shape)
    

        
if __name__ == "__main__":
    train()