import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import glob
import sys
import numpy as np

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])


class TwoDDataset(Dataset):
    def __init__(self, image_path, mask_path, transform_image=None, transform_mask=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self._add_images()

    def _add_images(self):
        self.images = glob.glob(self.image_path + "*.png")
        self.masks = glob.glob(self.mask_path + "*.txt")
        self.images.sort()
        self.masks.sort()
        self._check_valid_data(self.images, self.masks)

    def _check_valid_data(self, images, masks):
        images = [img[len(self.image_path):-4] for img in images]
        masks = [m[len(self.mask_path):-4] for m in masks]
        if images != masks:
            sys.exit('Invalid dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        image = io.imread(img_name)
        mask_file = open(mask_name,"r") 
        coordinate_list = mask_file.readlines()
        coordinate_list = [x.strip('\n').split(' ') for x in coordinate_list]
        coordinate_list = [[int(x[0]),int(x[1])] for x in coordinate_list]

        mask = np.zeros((image.shape[0],image.shape[1],1), dtype='uint8')
        for x,y in coordinate_list:
            mask[y][x] = [255]

        if self.transform_image: 
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask_file.close()

        return (image, mask)