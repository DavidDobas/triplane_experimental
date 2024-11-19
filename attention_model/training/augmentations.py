import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random

class AddGaussianNoise(object):
    """Add Gaussian noise to a tensor."""

    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class ElasticTransform(object):
    """Apply elastic deformation to a tensor."""
    def __init__(self, alpha=1, sigma=50):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        # Convert tensor to PIL Image
        img = TF.to_pil_image(img)
        # Apply elastic transformation using torchvision.transforms.functional
        # Note: Torchvision does not have elastic transform; consider using albumentations or implement manually
        # Placeholder: return the original image
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma})'