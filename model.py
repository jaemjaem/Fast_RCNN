from torchvision import models

import numpy as np
import torch
import torch.nn as nn

class Fast_RCNN(nn.Module):
    def __init__(self):
        super(Fast_RCNN, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.vgg16_feature = nn.Sequential(*(list(vgg16.features)[:-1])) # 마지막 max pooling 제거하고 Roi pooling으로다가
        self.vgg16_fc = nn.Sequential(*(list(vgg16.classifier)[:-1])) # 마지막 fc layer 제거하고 두개의 layer로다가

    def roi_pooling(self, features, batch_region):
        sub_sampling_ratio = features.shape[2] / 224.
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(7,7))

        result = []
        
        for batch_idx in range(len(batch_region)):
            outputs = []
            for region in batch_region[batch_idx]:
                x_min = int(np.around(region[0] * sub_sampling_ratio))
                y_min = int(np.around(region[1] * sub_sampling_ratio))
                x_max = int(np.around(region[2] * sub_sampling_ratio))
                y_max = int(np.around(region[3] * sub_sampling_ratio))
                sub_sampling_feature = features[batch_idx, :, x_min:x_max+1, y_min:y_max+1]
                output = self.max_pool(sub_sampling_feature)
                output = output.unsqueeze(0)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            #outputs = outputs.unsqueeze(0)
            result.append(outputs)
            
        result = torch.cat(result, dim=0)
        
        return result

    def forward(self, inputs, regions):
        x = self.vgg16_feature(inputs)
        x = self.roi_pooling(x, regions)
        x = x.view(x.size(0), -1) 
        x = self.vgg16_fc(x)

        return x