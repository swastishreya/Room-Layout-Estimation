import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import io

from MajdoorNet import MajdoorNet 
from utils import get_heatmap

"""Transforms"""
transform_image=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

transform_mask=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0],
                                        std=[1])])

"""Read test image to visualize"""
img = io.imread('/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/val/images/scene_00186_604421.png')
# mask_file = open('/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/val/labels/scene_00186_604421.txt',"r") 

"""Generate coordinate list"""
# coordinate_list = mask_file.readlines()
# coordinate_list = [x.strip('\n').split(' ') for x in coordinate_list]
# coordinate_list = [[int(x[0]),int(x[1])] for x in coordinate_list]

"""Get heatmap"""
# heatmap = get_heatmap(coordinate_list, img.shape)

"""Load mask"""
# f = open('val.npy', 'rb')
# loss_mask = torch.from_numpy(np.load(f))

"""Generate mask"""
# mask = np.zeros((img.shape[0],img.shape[1],1), dtype='uint8')
# for x,y in coordinate_list:
#     mask[y][x] = [255]

"""Transform image and mask"""
img = transform_image(img).unsqueeze(0)
# mask = transform_mask(mask).unsqueeze(0)

"""Get weighted loss"""
# pred = mask * loss_mask

"""Load model"""
majdoorNet = MajdoorNet(4, 1)
majdoorNet.load_state_dict(torch.load("model/majdoornet_heatmap_loss.pt"))
majdoorNet.eval()
outputs = majdoorNet(img)

outputs = outputs.squeeze().detach().numpy()

"""Display plot"""
plt.imshow(outputs)
plt.show()
