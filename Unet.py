import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, apply_activation=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.apply_activation = apply_activation
        
    def forward(self, x):
        """Output size is same as input size"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual

        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        residual = out
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        if self.apply_activation: out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1)
        self.rb = ResidualBlock(out_channels, out_channels, apply_activation=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        hidden = self.rb(out)
        out = self.max_pool(hidden)
        logging.debug("Encoder out size {}".format(out.size()))
        logging.debug("Encoder hidden size {}".format(hidden.size()))
        return hidden, out

class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
        self.rb = ResidualBlock(out_channels, out_channels, apply_activation=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hidden):
        logging.debug("Decoder input size {}".format(x.size()))
        out = self.upsample(x)
        out = self.conv1(out)
        logging.debug("Decoder out 1 size {}".format(out.size()))
        logging.debug("Decoder hidden size {}".format(hidden.size()))
        out = self.relu(out)
        out = out + hidden
        out = self.conv2(out)
        out = self.rb(out)
        logging.debug("Decoder out size {}".format(out.size()))
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1)
        self.rb = ResidualBlock(out_channels, out_channels, apply_activation=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.rb(out)
        logging.debug("Bottleneck out size {}".format(out.size()))
        return out

class Refinement(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, kernel_size=3):
        super(Refinement, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels+mid_channel, out_channels=out_channels, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        logging.debug("Upsampled: {}".format(upsampled.size()))
        logging.debug("bypass: {}".format(bypass.size()))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, hidden):
        out = self.conv1(x)
        out = self.relu(out)
        logging.debug("Refinement out 1 size {}".format(out.size()))
        logging.debug("Refinement hidden size {}".format(hidden.size()))
        out = self.crop_and_concat(hidden, out, crop=True)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        logging.basicConfig(level=logging.INFO)
        super(UNet, self).__init__()
        #Encode
        self.encoder1 = Encoder(in_channels=in_channel, out_channels=64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        # Bottleneck
        self.bottleneck = Bottleneck(256, 256)
        # Decode
        self.decoder3 = Decoder(256, 256, 128)
        self.decoder2 = Decoder(128, 128, 64)
        self.final_layer = Refinement(64, 64, out_channel)

    def forward(self, x):
        # Encode
        hidden_block1, encode_block1 = self.encoder1(x)
        hidden_block2, encode_block2 = self.encoder2(encode_block1)
        hidden_block3, encode_block3 = self.encoder3(encode_block2)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block3)
        # Decode
        decode_block3 = self.decoder3(bottleneck1, hidden_block3)
        decode_block2 = self.decoder2(decode_block3, hidden_block2)
        final_layer = self.final_layer(decode_block2, hidden_block1)
        return  final_layer