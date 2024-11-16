import os
import re
import datetime
import json

import torch
import torchmetrics
from torchsummary import summary
import torchvision
import torch.nn as nn

from training.trainable_module import TrainableModule
from training.dataset import ImageFolderDataset, PairwiseImageDataset, CombinedDataset
from training.loss import SSIMLoss, MSESSIMLoss, PerceptualLoss

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

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Ensure spatial dimensions match before addition
        if g1.shape[2:] != x1.shape[2:]:
            # Resize g1 to match x1's spatial dimensions
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        proj_key   = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x N
        energy     = torch.bmm(proj_query, proj_key)  # B x N x N
        attention  = self.softmax(energy)  # B x N x N
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class UNetWithAttentionAndSelfAttention(nn.Module):
    def __init__(self, num_narrowings=4, c_dim=25, channels_first=128, use_camera_in=True):
        super().__init__()
        
        self.c_dim = c_dim
        self.use_camera_in = use_camera_in
        if use_camera_in:
            self.init_conv = nn.Conv2d(1 + c_dim * 2, channels_first, kernel_size=3, padding=1)
        else:
            self.init_conv = nn.Conv2d(1 + c_dim, channels_first, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.self_attentions = nn.ModuleList()
        self.encoder_channels = []  # To track encoder output channels
        current_channels = channels_first
        for i in range(num_narrowings):
            out_channels = current_channels * 2
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            self.self_attentions.append(SelfAttention(out_channels))
            self.encoder_channels.append(out_channels)
            current_channels = out_channels
                
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        # Initialize Attention Gate and Self-Attention for decoder (num_narrowings times)
        self.attention_gates_decoder = nn.ModuleList()
        self.self_attentions_decoder = nn.ModuleList()
        for i in range(num_narrowings):
            in_channels = current_channels
            out_channels = current_channels // 2
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            if i < num_narrowings - 1:
                # Correct F_l to match encoder's skip connection channels
                F_g = out_channels
                F_l = self.encoder_channels[-(i + 2)]  # Corresponding encoder block
                self.attention_gates_decoder.append(AttentionGate(
                    F_g=F_g,
                    F_l=F_l,
                    F_int=F_g // 2
                ))
                self.self_attentions_decoder.append(SelfAttention(out_channels))
            current_channels = out_channels
                
        # Final convolution to obtain the image with 1 channel
        self.final_conv = nn.Conv2d(channels_first, 1, kernel_size=1)

    def forward(self, x, c_in, c_out):
        N, C, H, W = x.shape
        if self.use_camera_in:
            c_in_expanded = c_in.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            c_out_expanded = c_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            x = torch.cat([x, c_in_expanded, c_out_expanded], dim=1)
        else:
            c_out_expanded = c_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
            x = torch.cat([x, c_out_expanded], dim=1)
        
        x = self.init_conv(x)
        encoder_outputs = []
        
        # Encoder path
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            x = self.self_attentions[idx](x)  # Apply Self-Attention
            encoder_outputs.append(x)
        
        # Decoder path
        for idx, block in enumerate(self.decoder_blocks):
            x = block(x)
            if idx < len(self.attention_gates_decoder):
                # Retrieve the corresponding encoder skip connection
                skip_connection = encoder_outputs[-(idx + 2)]
                attention = self.attention_gates_decoder[idx](g=x, x=skip_connection)  # Apply Attention Gate
                x = attention  # Use the attention-modulated skip connection
                x = self.self_attentions_decoder[idx](x)  # Apply Self-Attention
                x = x + skip_connection  # Combine with skip connection
        
        x = self.final_conv(x)
        return x

class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, num_narrowings: int = 3, args: Args = None):
        super().__init__()
        # Custom
        self.unet = UNetWithAttentionAndSelfAttention(num_narrowings=num_narrowings, c_dim=c_dim, channels_first=args.unet_channels_first, use_camera_in=args.unet_use_camera_in)
    def forward(self, images, c_i, c_j):
        images = self.unet(images, c_i, c_j)
        if args.use_sigmoid:
            images = torch.sigmoid(images)
        else:
            images = images.clamp(0, 1)
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
    torchvision.utils.save_image(predictions[0][0:8], prediction_path)
    true_path = os.path.join(args.logdir, 'true.png')
    torchvision.utils.save_image(sample[0][0:8], true_path)

if __name__ == '__main__':
    args = Args(num_narrowings=4,
                loss='mse',
                unet_channels_first=128,
                unet_use_camera_in=False,
                dataset='datasets/chest_separate',
                pairwise_dataset_size=1000,
                batch_size=8,
                epochs=2,
                lr=0.0001,
                cosine_schedule=True,
                device='mps',
                use_sigmoid=True)
    main(args)
