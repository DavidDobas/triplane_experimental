import os
import re
import datetime
import json
import logging

import torch
import torchmetrics
import torchvision
import torch.nn as nn

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset, CombinedDataset
from training.loss import SSIMLoss, MSESSIMLoss, PerceptualLoss

from torchvision import models  # Import EfficientNet from torchvision
from torchvision.models import EfficientNet_B0_Weights
from torchsummary import summary


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.enable_logging = kwargs.get('enable_logging', False)  # Added logging argument


class DecoderBlock(nn.Module):
    """
    A single decoder block consisting of upsampling, concatenation with skip connections,
    and convolutional layers to process the combined features.
    """
    def __init__(self, in_channels, skip_channels, out_channels, dropout, enable_logging=False):
        super(DecoderBlock, self).__init__()
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.adjust_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x, skip):
        if self.enable_logging:
            self.logger.debug(f"  Before upsampling: {x.shape}")
        x = self.up(x)
        if self.enable_logging:
            self.logger.debug(f"  After upsampling: {x.shape}")

        # Ensure spatial dimensions match for concatenation
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            if self.enable_logging:
                self.logger.debug(f"  After interpolation: {x.shape}")

        if self.enable_logging:
            self.logger.debug(f"  Original Skip connection shape: {skip.shape}")
        skip = self.adjust_skip(skip)
        if self.enable_logging:
            self.logger.debug(f"  Adjusted Skip connection shape: {skip.shape}")

        x = torch.cat([x, skip], dim=1)
        if self.enable_logging:
            self.logger.debug(f"  After concatenation: {x.shape}")

        x = self.conv(x)
        if self.enable_logging:
            self.logger.debug(f"  After convolution: {x.shape}")
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, enable_logging=False):
        super(AttentionGate, self).__init__()
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        if self.enable_logging:
            self.logger.debug("  Computed attention weights.")
        return x * psi


class UNetWithAttention(nn.Module):
    def __init__(self, c_dim=25, use_camera_in=True, pretrained=True, enable_logging=False, dropout=0.2):
        super(UNetWithAttention, self).__init__()
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_camera_in = use_camera_in
        self.c_dim = c_dim

        # Load EfficientNet-B0 with appropriate weights
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.encoder = models.efficientnet_b0(weights=weights)
        self.encoder_features = list(self.encoder.features.children())
        self.skip_layers = [1, 3, 5, 7]  # Updated to match actual layer indices
        self.encoder_out_channels = [320, 112, 40, 16]  # Corrected to include 320 channels

        # Modify the first Conv2d layer to accept 1 input channel instead of 3
        original_conv = self.encoder.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        # Initialize new_conv's weights by averaging the original weights across the input channel dimension
        if pretrained and original_conv.weight.shape[1] == 3:
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.encoder.features[0][0] = new_conv

        # Integrate both camera in and out parameters after encoding
        if self.use_camera_in:
            self.camera_proj = nn.Linear(2 * self.c_dim, 1280)  # Assuming encoder's final output channels are 1280
        # Integrate only camera out parameter after encoding
        else:
            self.camera_proj = nn.Linear(self.c_dim, 1280)  # Assuming encoder's final output channels are 1280

        # Define decoder output channels
        self.decoder_out_channels = [112, 40, 16, 8]  # Corrected to match reversed encoder_out_channels

        # Initialize Decoder Blocks
        self.decoder_blocks = nn.ModuleList()
        current_channels = 1280  # Starting from the encoder's final output channels

        for out_channels, skip_channels in zip(self.decoder_out_channels, self.encoder_out_channels):
            decoder_block = DecoderBlock(
                in_channels=current_channels,     # From the encoder's final layer or previous decoder
                skip_channels=skip_channels,      # From encoder skip connection
                out_channels=out_channels,         # Desired output channels
                dropout=dropout,
                enable_logging=self.enable_logging
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = out_channels  # Update for the next decoder block

        # Upscale from 64x64 to 128x128
        self.upscale_block = nn.Sequential(
            nn.ConvTranspose2d(current_channels, current_channels//2, kernel_size=2, stride=2),
            nn.Conv2d(current_channels//2, current_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels//2),
            nn.ReLU(inplace=True)
        )
        current_channels = current_channels//2

        # Final convolution
        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=1)

    def forward(self, x, c_in=None, c_out=None):
        encoder_features = []
        for idx, layer in enumerate(self.encoder.features):
            x = layer(x)
            if idx in self.skip_layers:
                encoder_features.append(x)
                if self.enable_logging:
                    self.logger.debug(f"Collected encoder feature from layer {idx}")

        # Process and Integrate Camera Parameters After Encoding
        if self.use_camera_in:
            camera_params = torch.cat([c_in, c_out], dim=1)  # Shape: [N, 50] if c_dim=25
            camera_proj = self.camera_proj(camera_params).unsqueeze(-1).unsqueeze(-1)  # Shape: [N, 1280,1,1]
            x = x + camera_proj  # Broadcast addition
            if self.enable_logging:
                self.logger.debug(f"After camera integration, x shape: {x.shape}")
        else:
            camera_proj = self.camera_proj(c_out).unsqueeze(-1).unsqueeze(-1)  # Shape: [N, 1280,1,1]
            x = x + camera_proj  # Broadcast addition
            if self.enable_logging:
                self.logger.debug(f"After camera integration, x shape: {x.shape}")

        # Reverse the encoder_features to match decoder's expected order
        encoder_features = encoder_features[::-1]
        if self.enable_logging:
            self.logger.debug("Encoder features reversed for decoding.")

        # Debug: Check the lengths
        if self.enable_logging:
            self.logger.debug(f"Number of decoder blocks: {len(self.decoder_blocks)}")
            self.logger.debug(f"Number of encoder features (reversed): {len(encoder_features)}")

        # Ensure that the number of encoder_features matches the number of decoder_blocks
        if len(encoder_features) < len(self.decoder_blocks):
            raise IndexError(f"Insufficient encoder features for decoder blocks. "
                             f"Expected {len(self.decoder_blocks)} but got {len(encoder_features)}.")

        # Decoder pathway
        for idx, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = encoder_features[idx]  # Align decoder blocks with encoder features in reversed order
            if self.enable_logging:
                self.logger.debug(f"Decoder Block {idx+1}:")
                self.logger.debug(f"  Input Channels (before upsampling): {x.shape[1]}")
                self.logger.debug(f"  Skip Connection Channels: {skip_connection.shape[1]}")
            x = decoder_block(x, skip_connection)

        # Upscale from 64x64 to 128x128
        x = self.upscale_block(x)
        if self.enable_logging:
            self.logger.debug(f"  After upscale: {x.shape}")

        # Final Convolution
        x = self.final_conv(x)
        if self.enable_logging:
            self.logger.debug(f"Final Output shape: {x.shape}")
        return x


class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, args: Args = None):
        super().__init__()
        # Initialize UNet with Attention
        self.unet = UNetWithAttention(
            c_dim=c_dim,
            use_camera_in=args.unet_use_camera_in,
            pretrained=True,  # Use pretrained EfficientNet from torchvision
            enable_logging=args.enable_logging,  # Pass logging flag
            dropout=args.dropout
        )
        self.args = args
    
    def forward(self, images, c_i, c_j):
        # Forward pass through UNet
        images = self.unet(images, c_i, c_j)
        if self.args.use_sigmoid:
            images = torch.sigmoid(images)
        else:
            images = images.clamp(0, 1)
        return images


def main(args):
    model = Model(args=args)

    # Create logdir with timestamp
    timestamp = datetime.datetime.now()
    logdir = os.path.join(args.logdir, f"logdir_{timestamp.strftime('%Y%m%d')}_{timestamp.strftime('%H%M%S')}")
    # Save args to args.json in logdir
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # dataset_images = ImageFolderDataset(path=args.dataset, use_labels=True, max_size=None, xflip=False)
    # dataset_pairs = PairwiseImageDataset(dataset_images, size=args.pairwise_dataset_size)
    dataset_pairs_train = CombinedDataset(args.dataset, args.pairwise_dataset_size, augment=args.augment)
    dataset_pairs_dev = CombinedDataset(args.dataset_dev, args.pairwise_dataset_size, augment=False)

    generator = torch.Generator().manual_seed(42)

    train = torch.utils.data.DataLoader(dataset_pairs_train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dataset_pairs_dev, batch_size=args.batch_size)

    if args.loss == 'mse_ssim':
        loss = MSESSIMLoss()
    elif args.loss == 'perceptual':
        loss = PerceptualLoss()
    elif args.loss == 'mse':
        loss = torch.nn.MSELoss()
    elif args.loss == 'l1':
        loss = torch.nn.L1Loss()
    elif args.loss == 'mse_perceptual':
      mse_loss = torch.nn.MSELoss()
      perceptual_loss = PerceptualLoss()
      loss = lambda output, target: mse_loss(output, target) + perceptual_loss(output, target)
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
        logdir=logdir,
        device=args.device
    )

    logs = model.fit(dataloader=train, epochs=args.epochs, dev=dev)
    # Save training logs to logrid as logs.txt
    logs_path = os.path.join(logdir, 'logs.txt')
    with open(logs_path, 'w') as f:
        f.write(str(logs))
    
    # Save the first image from the dataset to logdir
    sample = next(iter(dev))[0]

    first_image = sample[0][1,:,:,:]  # Get the first image in the batch
    first_image_path = os.path.join(logdir, 'first_image.png')
    torchvision.utils.save_image(first_image, first_image_path)

    # Make a prediction on the first image and its camera
    model.eval()
    with torch.no_grad():
        predictions = model.predict(dev, as_numpy=False)
    prediction_path = os.path.join(logdir, 'prediction.png')
    torchvision.utils.save_image(predictions[0:8], prediction_path)
    true_path = os.path.join(logdir, 'true.png')
    torchvision.utils.save_image(sample[0][0:8], true_path)

    # Save the trained model
    # model_path = os.path.join(args.logdir, 'model.pt')
    # torch.save(model, model_path)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    
    # Model architecture
    parser.add_argument('--unet_use_camera_in', type=bool, default=False,
                        help='Whether to use camera parameters as input')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_sigmoid', type=bool, default=True,
                        help='Use sigmoid activation')

    # Training parameters
    parser.add_argument('--loss', type=str, default='mse_ssim',
                        choices=['mse', 'ssim', 'mse_ssim', 'mse_perceptual'],
                        help='Loss function to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--cosine_schedule', type=bool, default=True,
                        help='Use cosine learning rate schedule')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='datasets/chest_separate',
                        help='Path to training dataset')
    parser.add_argument('--dataset_dev', type=str, default='datasets/chest_separate_dev',
                        help='Path to validation dataset')
    parser.add_argument('--pairwise_dataset_size', type=int, default=None,
                        help='Size of pairwise dataset')
    parser.add_argument('--augment', type=bool, default=False,
                        help='Use data augmentation')

    # System parameters
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--enable_logging', type=bool, default=False,
                        help='Enable detailed logging')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory for saving logs and checkpoints')

    args = parser.parse_args()
    return Args(**vars(args))


if __name__ == '__main__':
    # args = Args(
    #     num_narrowings=6,
    #     loss='mse_ssim',
    #     unet_use_camera_in=False,  # Disabled input concatenation
    #     dataset='datasets/chest_separate',
    #     dataset_dev='datasets/chest_separate_dev',
    #     pairwise_dataset_size=None,
    #     batch_size=16,
    #     epochs=10,
    #     lr=0.0002,
    #     cosine_schedule=True,
    #     device='mps',  # Change to 'cuda' if available
    #     use_sigmoid=True,
    #     augment=False,
    #     enable_logging=False,  # Logging disabled by default
    #     dropout=0.2
    # )
    args = parse_args()
    main(args)