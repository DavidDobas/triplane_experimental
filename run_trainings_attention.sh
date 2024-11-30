cd attention_model
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0 --unet_input_dropout=0
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0.1 --unet_input_dropout=0.1
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0.2 --unet_input_dropout=0.1
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0.3 --unet_input_dropout=0.1
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0.4 --unet_input_dropout=0.1
python train.py --loss=mse_ssim --batch_size=16 --epochs=4 --lr=0.0002 --cosine_schedule=True --augment=False --unet_dropout=0.5 --unet_input_dropout=0.1