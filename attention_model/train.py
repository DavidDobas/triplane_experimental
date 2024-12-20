import os
import re
import datetime
import json
import argparse
import torch
import torchmetrics
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
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class UNetWithAttention(nn.Module):
    def __init__(self, num_narrowings=3, c_dim=25, channels_first=128, use_camera_in=True, dropout=0.1, input_dropout=0.1):
        super().__init__()
        
        self.c_dim = c_dim
        self.use_camera_in = use_camera_in
        if use_camera_in:
            self.init_conv = nn.Conv2d(1 + c_dim * 2, channels_first, kernel_size=3, padding=1)
        else:
            self.init_conv = nn.Conv2d(1 + c_dim, channels_first, kernel_size=3, padding=1)
        
        # Dropout layer for input
        self.input_dropout = nn.Dropout(p=input_dropout)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_channels = channels_first
        for i in range(num_narrowings):
            out_channels = current_channels * 2
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
            current_channels = out_channels
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
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
            # Correct Attention Gate Initialization
            self.attention_gates.append(AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2))
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

        x = self.input_dropout(x)
        
        x = self.init_conv(x)
        encoder_outputs = []
        
        # Encoder path with sparsity
        for idx, block in enumerate(self.encoder_blocks):
            encoder_outputs.append(x)
            x = block(x)
        
        # Decoder path with attention
        for idx, block in enumerate(self.decoder_blocks):
            x = block(x)
            skip_connection = encoder_outputs[-(idx + 1)]
            attn_gate = self.attention_gates[idx]
            skip_attention = attn_gate(g=x, x=skip_connection)
            # Element-wise addition to combine decoder features with attended encoder features
            x = x + skip_attention  # Ensure channel dimensions match after addition
    
        x = self.final_conv(x)
        return x

class Model(TrainableModule):
    def __init__(self, c_dim: int = 25, num_narrowings: int = 3, args: Args = None):
        super().__init__()
        # Custom
        self.unet = UNetWithAttention(num_narrowings=num_narrowings, c_dim=c_dim, channels_first=args.unet_channels_first, use_camera_in=args.unet_use_camera_in, dropout=args.unet_dropout, input_dropout=args.unet_input_dropout)
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

    # Save the model
    # model_path = os.path.join(args.logdir, 'model.pt')
    # torch.save(model, model_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet with Attention')
    parser.add_argument('--num_narrowings', type=int, default=6,
                        help='Number of narrowing layers in UNet')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'ssim', 'mse_ssim', 'perceptual', 'mse_perceptual', 'l1'],
                        help='Loss function to use')
    parser.add_argument('--unet_channels_first', type=int, default=128,
                        help='Number of channels in first UNet layer')
    parser.add_argument('--unet_use_camera_in', type=bool, default=False,
                        help='Use camera input in UNet')
    parser.add_argument('--dataset', type=str, default='datasets/chest_separate',
                        help='Path to training dataset')
    parser.add_argument('--dataset_dev', type=str, default='datasets/chest_separate_dev',
                        help='Path to validation dataset')
    parser.add_argument('--pairwise_dataset_size', type=int, default=None,
                        help='Size of pairwise dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--cosine_schedule', type=bool, default=True,
                        help='Use cosine learning rate schedule')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--use_sigmoid', type=bool, default=True,
                        help='Use sigmoid activation')
    parser.add_argument('--augment', type=bool, default=False,
                        help='Use data augmentation')
    parser.add_argument('--unet_dropout', type=float, default=0,
                        help='Dropout rate in UNet')
    parser.add_argument('--unet_input_dropout', type=float, default=0,
                        help='Input dropout rate in UNet')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory for saving logs')
                        
    args = parser.parse_args()
    return Args(**vars(args))

if __name__ == '__main__':
    # args = Args(num_narrowings=6,
    #             loss='mse',
    #             unet_channels_first=128,
    #             unet_use_camera_in=False,
    #             dataset='datasets/chest_separate',
    #             dataset_dev='datasets/chest_separate_dev',
    #             pairwise_dataset_size=None,
    #             batch_size=16,
    #             epochs=10,
    #             lr=0.0002,
    #             cosine_schedule=True,
    #             device='cuda',
    #             use_sigmoid=True,
    #             augment=False,
    #             unet_dropout=0.1,
    #             unet_input_dropout=0.5)
    args = parse_args()
    main(args)
