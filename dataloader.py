import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import glob

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 


class DocumentDeblurrDataset(Dataset):
    def __init__(self, blurred_image_path, real_image_path, transform=None):
        self.blurred_image_path = blurred_image_path
        self.real_image_path = real_image_path
        self.transform = transform
        self._add_images()

    def _add_images(self):
        self.blurred_images = glob.glob(self.blurred_image_path + "n_*/*.png")
        set_of_blurred_images = len(self.blurred_images)
        self.real_images = glob.glob(self.real_image_path + "*.png")
        total_ground_truth = len(self.real_images)
        self.real_images = self.real_images*int((set_of_blurred_images/total_ground_truth))

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        blurred_img_name = self.blurred_images[idx]
        real_image_name = self.real_images[idx]
        blurred_image = io.imread(blurred_img_name)
        real_image = io.imread(real_image_name)

        if self.transform: 
            blurred_image = self.transform(blurred_image)
            real_image = self.transform(real_image)

        return (blurred_image, real_image)


if __name__ == "__main__":
    d = DocumentDeblurrDataset("/home/swasti/Documents/PE/BMVC_image_quality_train_data/",
                               "/home/swasti/Documents/PE/BMVC_image_quality_train_data/orig/",
                               transform)
    print(len(d))