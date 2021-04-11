from torch.utils.data import DataLoader
from dataset import Fast_RCNN_Dataset
from model import Fast_RCNN
import torch.optim as optim

import torch
import torch.nn as nn
import torchvision.transforms as transforms

image_path = "./VOC2012/image/"
label_path = "./VOC2012/label/"

weight_save_path = './checkpoints/'

log_criterion = nn.CrossEntropyLoss()
smooth_l1_criterion = nn.SmoothL1Loss()

N = 64 # num_sampling
K = 1 # num_classes

epochs = 300

def multi_task_loss(target_class, target_bbox, target_mask, classifier_vector, bbox_vector):
    lambda_value = 1.
    target_class = target_class.view(-1, target_class.size(2)) # target class 와 output class score vector 랑 크기 맞추기
    target_bbox = target_bbox.view(-1, K + 1, target_bbox.size(2))  # target bbox 와 output bbox vector 랑 크기 맞추기
    target_mask = target_mask.view(-1, K + 1, target_mask.size(3))

    # target class 와 classifier vector는 그냥 무지성으로 log loss 쓰면 될듯
    # target bbox 와 bbox vector 는 mask 를 이용해서 무지성 smoothl1loss
    log_loss = log_criterion(classifier_vector, torch.max(target_class, 1)[1])
    bbox_loss = smooth_l1_criterion(bbox_vector * target_mask, target_bbox * target_mask)

    return log_loss + lambda_value * bbox_loss

def train():
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Fast_RCNN_Dataset(image_path, label_path, N, K, transform)
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2)

    model = Fast_RCNN().cuda()
    model.train(True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            img, target_class, target_bbox, target_mask, positive_sampling, negative_sampling = batch
            optimizer.zero_grad()

            img = img.cuda()
            target_class = target_class.cuda()
            target_bbox = target_bbox.cuda()
            target_mask = target_mask.cuda()
            sampling = torch.cat([positive_sampling, negative_sampling], dim=1)

            classifier_vector, bbox_vector = model(img, sampling)
            loss = multi_task_loss(target_class, target_bbox, target_mask, classifier_vector, bbox_vector)
            loss.backward()
            optimizer.step()

            running_loss += loss

            if i == 1 : # batch = 1 일때, iteration이 2개임 ex) 0, 1 그래서 그냥 1일때 출력..
                print("epoch : %d/%d, iteration : %d, loss : %.3f" % (epoch + 1, epochs, i, running_loss))
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), weight_save_path + "fast_rcnn_%d.pth" % (epoch + 1))
        
if __name__ == "__main__":
    train()