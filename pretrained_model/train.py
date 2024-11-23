import os
import re
import datetime
import json

import torch
import torchmetrics
import torchvision
import torch.nn as nn

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset, CombinedDataset
from training.loss import SSIMLoss, MSESSIMLoss, PerceptualLoss

from torchvision import models  # Import EfficientNet from torchvision
from torchsummary import summary


class DecoderBlock(nn.Module):
    """
    A single decoder block consisting of upsampling, concatenation with skip connections,
    and convolutional layers to process the combined features.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        print(f"  Before upsampling: {x.shape}")
        x = self.up(x)
        print(f"  After upsampling: {x.shape}")

        # Ensure spatial dimensions match for concatenation
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            print(f"  After interpolation: {x.shape}")

        print(f"  Skip connection shape: {skip.shape}")
        x = torch.cat([x, skip], dim=1)
        print(f"  After concatenation: {x.shape}")

        x = self.conv(x)
        print(f"  After convolution: {x.shape}")
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
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
        # g: gating signal (from decoder)
        # x: skip connection feature map (from encoder)

        print(f"  AttentionGate - Before Processing: g shape: {g.shape}, x shape: {x.shape}")
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        print(f"  AttentionGate - Psi shape: {psi.shape}")
        return x * psi


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class UNetWithAttention(nn.Module):
    def __init__(self, num_narrowings=3, c_dim=25, use_camera_in=True, pretrained=True):
        super().__init__()
        
        self.c_dim = c_dim
        self.use_camera_in = use_camera_in

        # Input Channels: Only the original single-channel image
        input_channels = 1  # Since camera parameters are added after encoding

        # Initialize EfficientNet as the encoder from torchvision
        self.encoder = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify the first convolution layer to accept single-channel input
        original_conv = self.encoder.features[0][0]  # Typically, features[0][0] is Conv2d
        new_conv = nn.Conv2d(
            input_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias
        )
        if pretrained:
            with torch.no_grad():
                # Initialize new_conv weights by averaging the original weights across the RGB channels
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        self.encoder.features[0][0] = new_conv
        
        # Optionally freeze the encoder weights
        if pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Define the channels from EfficientNet for skip connections
        self.encoder_out_channels = [16, 24, 40, 80, 112, 1280]  # Aligned with skip_layers
        self.skip_layers = [1, 3, 5, 7, 13, 16]  # Indices in self.encoder.features
        
        # Project Camera Parameters to Match Encoder's Deepest Channel Dimension (1280)
        self.camera_proj = nn.Linear(c_dim * 2, self.encoder_out_channels[-1])
        
        # Initialize Decoder Blocks
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        current_channels = self.encoder_out_channels[-1]  # Start with the deepest layer (1280)
        
        # Number of Decoder Steps
        num_decoder_steps = len(self.skip_layers) - 1  # 5 for EfficientNet-B0
        
        # Define Decoder Blocks
        for idx in reversed(range(num_decoder_steps)):
            skip_layer_idx = self.skip_layers[idx]
            skip_channels = self.encoder_out_channels[idx]
            out_channels = self.encoder_out_channels[idx]
            
            decoder_block = DecoderBlock(
                in_channels=current_channels,    # From previous decoder block
                skip_channels=skip_channels,     # From encoder skip connection
                out_channels=out_channels        # Desired output channels
            )
            self.decoder_blocks.append(decoder_block)
            current_channels = out_channels  # Update for next block
        
        # Final Convolution Layer to Obtain Single-Channel Output
        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=1)
    
    def forward(self, x, c_in, c_out):
        N, C, H, W = x.shape
        # Pass x to encoder without any additional channels.

        # 1. Encoder Forward Pass
        encoder_features = []
        for idx, layer in enumerate(self.encoder.features):
            x = layer(x)
            if idx in self.skip_layers:
                encoder_features.append(x)
        
        # 2. Process and Integrate Camera Parameters After Encoding
        camera_params = torch.cat([c_in, c_out], dim=1)  # Shape: [N, 50] if c_dim=25
        camera_proj = self.camera_proj(camera_params)      # Shape: [N, 1280]
        camera_proj = camera_proj.unsqueeze(2).unsqueeze(3)  # Shape: [N, 1280, 1, 1]
        # Expand to match spatial dimensions
        camera_proj = camera_proj.expand(-1, -1, x.size(2), x.size(3))  # Shape: [N, 1280, H', W']
        # Integrate camera information with encoder features
        x = x + camera_proj
        print(f"After camera integration, x shape: {x.shape}")  # Debugging

        # 3. Decoder Path with Attention
        for idx, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = encoder_features[-(idx + 2)]  # Get corresponding skip connection
            
            # Debugging: Print channel numbers
            print(f"Decoder Block {idx+1}:")
            print(f"  Input Channels (before upsampling): {x.shape[1]}")
            print(f"  Skip Connection Channels: {skip_connection.shape[1]}")
            
            x = decoder_block(x, skip_connection)
        
        # 4. Final Convolution Layer
        x = self.final_conv(x)  # Shape: [N, 1, H, W]
        return x


class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, num_narrowings: int = 3, args: Args = None):
        super().__init__()
        # Initialize UNet with Attention
        self.unet = UNetWithAttention(
            num_narrowings=num_narrowings,
            c_dim=c_dim,
            use_camera_in=args.unet_use_camera_in,
            pretrained=True  # Use pretrained EfficientNet from torchvision
        )
    
    def forward(self, images, c_i, c_j):
        # Forward pass through UNet
        images = self.unet(images, c_i, c_j)
        if self.args.use_sigmoid:
            images = torch.sigmoid(images)
        else:
            images = images.clamp(0, 1)
        return images


def main(args):
    model = Model(num_narrowings=args.num_narrowings, args=args)

    # Create log directory with timestamp
    timestamp = datetime.datetime.now()
    args.logdir = f"logdir_{timestamp.strftime('%Y%m%d')}_{timestamp.strftime('%H%M%S')}"
    # Save args to args.json in logdir
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Initialize datasets
    dataset_pairs_train = CombinedDataset(args.dataset, args.pairwise_dataset_size, augment=args.augment)
    dataset_pairs_dev = CombinedDataset(args.dataset_dev, args.pairwise_dataset_size, augment=False)

    generator = torch.Generator().manual_seed(42)

    train = torch.utils.data.DataLoader(dataset_pairs_train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dataset_pairs_dev, batch_size=args.batch_size)

    # Select loss function
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

    # Learning rate scheduler
    if args.cosine_schedule:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train), eta_min=0)
    else:
        schedule = None

    # Configure the model for training
    model.configure(
        optimizer=optimizer,
        schedule=schedule,
        loss=loss,
        metrics=[torchmetrics.image.PeakSignalNoiseRatio()],
        logdir=args.logdir,
        device=args.device
    )

    # Start training
    model.fit(dataloader=train, epochs=args.epochs, dev=dev)
    
    # Save and log outputs
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
    torchvision.utils.save_image(predictions[0][0:8], prediction_path)
    true_path = os.path.join(args.logdir, 'true.png')
    torchvision.utils.save_image(sample[0][0:8], true_path)

    # Save the trained model
    model_path = os.path.join(args.logdir, 'model.pt')
    torch.save(model, model_path)


if __name__ == '__main__':
    args = Args(
        num_narrowings=6,
        loss='mse',
        unet_use_camera_in=False,  # Disabled input concatenation
        dataset='datasets/chest_separate',
        dataset_dev='datasets/chest_separate_dev',
        pairwise_dataset_size=None,
        batch_size=16,
        epochs=10,
        lr=0.0002,
        cosine_schedule=True,
        device='cpu',  # Change to 'cuda' if available
        use_sigmoid=True,
        augment=True
    )
    main(args)