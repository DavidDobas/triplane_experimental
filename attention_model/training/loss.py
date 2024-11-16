import torch
import torchmetrics
import torch.nn as nn
import torchvision.models as models
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, predicted, target):
        ssim_score = self.ssim(predicted, target)
        return 1 - ssim_score  # Convert similarity to loss
    
class MSESSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, channel=1):
        super(MSESSIMLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.ssim = SSIMLoss(data_range=data_range, channel=channel)

    def forward(self, predicted, target):
        mse_loss = self.mse(predicted, target)
        ssim_loss = self.ssim(predicted, target)
        return mse_loss + ssim_loss

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=34, use_cuda=True):
        super(PerceptualLoss, self).__init__()
        self.use_cuda = use_cuda
        # Load pre-trained VGG19 model
        vgg = models.vgg19(pretrained=True).features
        # Modify the first convolutional layer to accept 1 channel
        vgg[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        if self.use_cuda:
            vgg = vgg.cuda()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg[:feature_layer + 1]
        self.criterion = nn.L1Loss()

    def forward(self, generated, target):
        # Ensure images are single-channel
        if generated.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input images must be single-channel (grayscale)")
        
        # Pass through VGG
        gen_features = self.vgg(generated)
        tgt_features = self.vgg(target)
        # Compute perceptual loss
        loss = self.criterion(gen_features, tgt_features)
        return loss
    
    # training/losses.py
