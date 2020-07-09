from Unet import UNet
from dataloader import DocumentDeblurrDataset
from metrics import PSNR, SSIM

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from collections import defaultdict
from tqdm import tqdm, trange
import gc
import copy
import time

class Trainer:

    def __init__(self, model, optimizer, criterion, scheduler, num_epochs=25, batch_size=5):
        gc.collect()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.use_gpu = torch.cuda.is_available()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if self.use_gpu:
            self.model.cuda()
        self.num_epochs = num_epochs
        self.load_dataset(batch_size)

    def load_dataset(self, batch_size):
        train_set = DocumentDeblurrDataset("/home/swasti/Documents/PE/BMVC_image_quality_train_data/",
                                         "/home/swasti/Documents/PE/BMVC_image_quality_train_data/orig/",
                                         transform=self.transform)
        val_set = DocumentDeblurrDataset("/home/swasti/Documents/PE/BMVC_image_quality_val_data/",
                                         "/home/swasti/Documents/PE/BMVC_image_quality_val_data/orig/",
                                         transform=self.transform)
        self.image_datasets = {
            'train': train_set, 'val': val_set
        }

        self.dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }

    def calculate_loss(self, outputs, real_images, criterion, metrics):
        loss = self.criterion(outputs, real_images)
        # metrics['PSNR'] = PSNR(outputs, real_images)
        metrics['SSIM'] += SSIM(outputs, real_images).data.cpu().numpy() * real_images.size(0)
        metrics['Loss'] += loss.data.cpu().numpy() * real_images.size(0)
        return loss


    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for blurred_images, real_images in self.dataloaders[phase]:
                    if self.use_gpu:
                        blurred_images = blurred_images.cuda()
                        real_images = real_images.cuda()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(blurred_images)
                        loss = self.calculate_loss(outputs, real_images, self.criterion, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    epoch_samples += blurred_images.size(0)

                if phase == 'train':
                    self.scheduler.step()

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        gc.collect()

if __name__ == "__main__":

    unet = UNet(in_channel=3,out_channel=3)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    trainer = Trainer(unet,optimizer, criterion, exp_lr_scheduler)
    trainer.train()
