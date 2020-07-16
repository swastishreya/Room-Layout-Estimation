import torch
import torch.nn as nn

import numpy as np
import time

def simple_distance_loss(pred, target, coordinate_list, use_gpu, info, umbrella_size=100):
    loss_mask = np.zeros(target.shape, dtype='uint8')

    if info['epoch'] == 0:
        for target_y_cor, target_x_cor in coordinate_list:
            for x_cor in range(max(0,target_x_cor-umbrella_size),min(512,target_x_cor+umbrella_size)):
                for y_cor in range(max(0,target_y_cor-umbrella_size),min(1024,target_y_cor+umbrella_size)):
                    if x_cor != target_x_cor and y_cor != target_y_cor:
                        weight = 1./(abs(x_cor - target_x_cor) + abs(y_cor - target_y_cor))
                        if weight > loss_mask[0][0][x_cor][y_cor]:
                            loss_mask[0][0][x_cor][y_cor] = weight
                    else:
                        loss_mask[0][0][x_cor][y_cor] = 1

        f = open(info['phase']+'.npy', 'ab')
        np.save(f, loss_mask)
        f.close()

    else:
        loss_mask = np.load(info['array_file'])

    loss_mask = torch.from_numpy(loss_mask) 

    if use_gpu:
        loss_mask = loss_mask.cuda()
    
    weighted_loss = (1 - (pred * loss_mask)).mean()
    weighted_loss = weighted_loss/len(coordinate_list)

    return weighted_loss

def naive_distance_loss(pred, target, coordinate_list, use_gpu, umbrella_size=100):
    loss_mask = torch.from_numpy(np.zeros(target.shape, dtype='uint8'))

    # start = time.time()

    for x_cor in range(target.shape[2]):
        for y_cor in range(target.shape[3]):
            for target_y_cor, target_x_cor in coordinate_list:
                if x_cor != target_x_cor and y_cor != target_y_cor:
                    weight = 1./(abs(x_cor - target_x_cor) + abs(y_cor - target_y_cor))
                    if weight > loss_mask[0][0][x_cor][y_cor]:
                        loss_mask[0][0][x_cor][y_cor] = weight
                else:
                    loss_mask[0][0][x_cor][y_cor] = 1

    # print("Done with loss_mask generation: {}".format(time.time() - start))

    if use_gpu:
        loss_mask = loss_mask.cuda()
    
    weighted_loss = (pred * loss_mask).mean()
    weighted_loss = weighted_loss/len(coordinate_list)

    return weighted_loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()
