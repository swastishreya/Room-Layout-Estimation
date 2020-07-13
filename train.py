# from Unet import UNet
from MajdoorNet import MajdoorNet
from dataloader import TwoDDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import dice_loss, simple_distance_loss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from collections import defaultdict
from tqdm import tqdm, trange
import gc
import copy
import time

class Trainer:

    def __init__(self, model, optimizer, scheduler, num_epochs=25, batch_size=1):
        gc.collect()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.num_epochs = num_epochs
        self.load_dataset(batch_size)

    def load_dataset(self, batch_size):
        train_set = TwoDDataset("/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/train/images/",
                                "/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/train/labels/",
                                transform_image=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]),
                                transform_mask=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0],
                                                    std=[1])]))
        val_set = TwoDDataset("/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/val/images/",
                              "/home/swasti/Documents/hackathon/Room-Layout-Estimation/data/val/labels/",
                              transform_image=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]),
                              transform_mask=transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0],
                                                                    std=[1])]))
        self.image_datasets = {
            'train': train_set, 'val': val_set
        }

        self.dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=12)
        }

    def calculate_loss(self, pred, target, coordinate_list, metrics, info, bce_weight=0):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = torch.sigmoid(pred)
        # dice = dice_loss(pred, target)

        dist_loss = simple_distance_loss(pred, target, coordinate_list, self.use_gpu, info)

        loss = bce * bce_weight + dist_loss * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dist_loss'] += dist_loss.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        return loss


    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        PATH = "model/majdoornet_dist_loss.pt"

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

                info = { 'phase': phase, 'epoch': epoch }
                if epoch != 0:
                    f = open(info['phase']+'.npy', 'rb')
                    info['array_file'] = f

                for images, masks, coordinate_list in tqdm(self.dataloaders[phase]):
                    if self.use_gpu:
                        images = images.cuda()
                        masks = masks.cuda()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)
                        loss = self.calculate_loss(outputs, masks, coordinate_list, metrics, info)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    epoch_samples += images.size(0)

                if phase == 'train':
                    self.scheduler.step()

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), PATH)

                if epoch != 0:
                    info['array_file'].close()
                    
            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        gc.collect()

if __name__ == "__main__":

    # unet = UNet(in_channel=4,out_channel=1)
    majdoorNet = MajdoorNet(4, 1)
    # criterion = nn.L1Loss() # doesn't matter here 
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, majdoorNet.parameters()), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    trainer = Trainer(majdoorNet, optimizer, exp_lr_scheduler)
    trainer.train()
