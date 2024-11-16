import os
import re
import datetime
import json

import torch
import torchmetrics
import torchvision

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset, CombinedDataset
from training.loss import SSIMLoss, MSESSIMLoss, PerceptualLoss

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class UNet(torch.nn.Module):
    def __init__(self, num_narrowings=3, c_dim=25, channels_first=128, use_camera_in=True):
        super().__init__()
        
        self.c_dim = c_dim
        self.use_camera_in = use_camera_in
        # Initial convolution to get to channels_first channels
        if use_camera_in:
            self.init_conv = torch.nn.Conv2d(1 + c_dim * 2, channels_first , kernel_size=3, padding=1)  # Updated input channels
        else:
            self.init_conv = torch.nn.Conv2d(1 + c_dim, channels_first , kernel_size=3, padding=1)  # Updated input channels
        
        # Encoder blocks
        self.encoder_blocks = torch.nn.ModuleList()
        current_channels = channels_first
        for i in range(num_narrowings):
            out_channels = current_channels * 2
            self.encoder_blocks.append(torch.nn.Sequential(
                torch.nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ))
            current_channels = out_channels
            
        # Decoder blocks
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(num_narrowings):
            in_channels = current_channels
            out_channels = current_channels // 2
            self.decoder_blocks.append(torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ))
            current_channels = out_channels
            
        # Final convolution to obatin the image with 1 channel
        self.final_conv = torch.nn.Conv2d(channels_first, 1, kernel_size=1)
    
    def forward(self, x, c_in, c_out):
        # Concatenate camera parameters to the input
        # Assuming c_in and c_out are of shape [N, c_dim] and needs to be expanded spatially
        N, C, H, W = x.shape
        if self.use_camera_in:
            c_in_expanded = c_in.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            c_out_expanded = c_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            x = torch.cat([x, c_in_expanded, c_out_expanded], dim=1)  # Shape: [N, 1 + c_dim, H, W]
        else:
            c_out_expanded = c_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            x = torch.cat([x, c_out_expanded], dim=1)  # Shape: [N, 1 + c_dim, H, W]
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for block in self.encoder_blocks:
            encoder_outputs.append(x)
            x = block(x)
            
        # Decoder path with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(encoder_outputs)):
            x = block(x)
            # Add skip connection
            x = x + skip
            
        # Final 1x1 convolution
        x = self.final_conv(x)
        
        return x
    

class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, num_narrowings: int = 3, args: Args = None):
        super().__init__()
        # Custom
        self.unet = UNet(num_narrowings=num_narrowings, c_dim=c_dim, channels_first=args.unet_channels_first, use_camera_in=args.unet_use_camera_in)
    def forward(self, images, c_i, c_j):
        images = self.unet(images, c_i, c_j)
        images = torch.sigmoid(images)
        return images


def main(args):
    model = Model(num_narrowings=args.num_narrowings, args=args)

    # Create logdir with timestamp
    timestamp = datetime.datetime.now()
    args.logdir = f"logdir_{timestamp.strftime('%Y%m%d')}_{timestamp.strftime('%H%M%S')}"
    # Save args to args.json in logdir
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # dataset_images = ImageFolderDataset(path=args.dataset, use_labels=True, max_size=None, xflip=False)
    # dataset_pairs = PairwiseImageDataset(dataset_images, size=args.pairwise_dataset_size)
    dataset_pairs = CombinedDataset(args.dataset, args.pairwise_dataset_size)

    generator = torch.Generator().manual_seed(42)
    train, dev = torch.utils.data.random_split(dataset_pairs, [int(0.9*len(dataset_pairs)), len(dataset_pairs) - int(0.9*len(dataset_pairs))], generator=generator)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    if args.loss == 'mse_ssim':
        loss = MSESSIMLoss()
    elif args.loss == 'perceptual':
        loss = PerceptualLoss()
    elif args.loss == 'mse':
        loss = torch.nn.MSELoss()
    elif args.loss == 'l1':
        loss = torch.nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.cosine_schedule:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train), eta_min=0)
    else:
        schedule=None

    model.configure(
        optimizer=optimizer,
        schedule=schedule,
        loss=loss,
        metrics=[torchmetrics.image.PeakSignalNoiseRatio()],
        logdir=args.logdir,
        device=args.device
    )

    model.fit(dataloader=train, epochs=args.epochs, dev=dev)
    
    # Save the first image from the dataset to logdir
    sample = next(iter(dev))[0]

    first_image = sample[0][1,:,:,:]  # Get the first image in the batch
    first_image_path = os.path.join(args.logdir, 'first_image.png')
    torchvision.utils.save_image(first_image, first_image_path)

    # Make a prediction on the first image and its camera
    model.eval()
    with torch.no_grad():
        predictions = model.predict(dev, as_numpy=False)
    prediction_path = os.path.join(args.logdir, 'prediction.png')
    true_path = os.path.join(args.logdir, 'true.png')
    torchvision.utils.save_image(predictions[0][0], prediction_path)
    torchvision.utils.save_image(sample[0][0], true_path)

if __name__ == '__main__':
    args = Args(num_narrowings=4,
                loss='mse_ssim',
                unet_channels_first=128,
                unet_use_camera_in=False,
                dataset='datasets/chest_separate',
                pairwise_dataset_size=2000,
                batch_size=16,
                epochs=3,
                lr=0.0001,
                cosine_schedule=True,
                device='mps')
    main(args)
