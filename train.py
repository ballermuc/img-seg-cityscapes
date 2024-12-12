import argparse
import os
import subprocess
from datetime import datetime
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
import wandb
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss  
from dataset_cityscapes import *
from epoch import *

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")

    # General training parameters
    parser.add_argument('--batch_size_training', type=int, required=False, default=8, help="Batch size for training")
    parser.add_argument('--batch_size_validation', type=int, required=False, default=4, help="Batch size for validation")
    parser.add_argument('--batch_size_test', type=int, required=False, default=1, help="Batch size for testing")
    parser.add_argument('--epochs', type=int, required=False, default=20, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], required=False, default='Adam', help="Optimizer to use: 'Adam' or 'SGD'")
    parser.add_argument('--num_workers', type=int, required=False, default=16, help="Number of workers for DataLoader")

    # Model-specific parameters
    parser.add_argument('--model', type=str, choices=['DeepLabV3P+', 'Unet'], required=False, default='DeepLabV3P+', help="Model to use: 'DeepLabV3P+' or 'Unet'")
    parser.add_argument('--encoder', type=str, required=False, default='efficientnet-b4', help="Encoder/backbone model name")
    parser.add_argument('--encoder_weights', type=str, required=False, default="imagenet", help="Pre-trained weights for the encoder: 'imagenet' or 'None'")
    parser.add_argument('--patch_size', type=int, required=False, default=512, help="Input image patch size (height and width must be divisible by 16)")

    # Loss functions
    parser.add_argument('--loss', type=str, choices=['ce', 'dice', 'focal'], required=False, default='ce', help="Loss function: 'ce' for CrossEntropy, 'dice', or 'focal'")
    parser.add_argument('--class_weights', type=str, required=False, default=None, help="Class weights for loss function (e.g., '11:10,12:10') or 'none'")

    # DeepLabV3+ specific parameters
    parser.add_argument('--encoder_output_stride', type=int, choices=[8, 16], required=False, default=16)
    parser.add_argument('--decoder_atrous_rates', type=str, required=False, default="12,24,36")
    parser.add_argument('--decoder_channels', type=int, required=False, default=256)
    parser.add_argument('--upsampling', type=int, required=False, default=4)

    # Unet specific parameters
    parser.add_argument('--decoder_use_batchnorm', type=str, choices=['True', 'False', 'inplace'], required=False, default='True')
    parser.add_argument('--decoder_attention_type', type=str, choices=['None', 'scse'], required=False, default=None)

    return parser.parse_args()

def print_configuration(config):
    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    
    # Model and training basics
    print(f"{'Model':25}: {config.model}")
    print(f"{'Encoder':25}: {config.encoder}")
    print(f"{'Encoder Weights':25}: {config.encoder_weights}")
    print(f"{'Epochs':25}: {config.epochs}")
    print(f"{'Batch Size (Training)':25}: {config.batch_size_training}")
    print(f"{'Batch Size (Validation)':25}: {config.batch_size_validation}")
    print(f"{'Batch Size (Test)':25}: {config.batch_size_test}")
    print(f"{'Learning Rate':25}: {config.learning_rate}")
    print(f"{'Optimizer':25}: {config.optimizer}")
    print(f"{'Number of Workers':25}: {config.num_workers}")

    # Patch size and loss function
    print("\n" + "-"*50)
    print(f"{'Patch Size':25}: {config.patch_size}")
    print(f"{'Loss Function':25}: {config.loss}")
    print(f"{'Class Weights':25}: {config.class_weights}")

    # Model-specific parameters
    print("\n" + "-"*50)
    print("Model-Specific Parameters")
    print("-"*50)
    if config.model == "DeepLabV3P+":
        print(f"{'Encoder Output Stride':25}: {config.encoder_output_stride}")
        print(f"{'Decoder Atrous Rates':25}: {config.decoder_atrous_rates}")
        print(f"{'Decoder Channels':25}: {config.decoder_channels}")
        print(f"{'Upsampling':25}: {config.upsampling}")
    elif config.model == "Unet":
        print(f"{'Decoder Use BatchNorm':25}: {config.decoder_use_batchnorm}")
        print(f"{'Decoder Attention Type':25}: {config.decoder_attention_type}")
    
    print("="*50 + "\n")


def parse_class_weights(class_weights_arg, num_classes, device):
    if class_weights_arg is None or class_weights_arg.lower() == "none":
        return None
    class_weights = torch.ones(num_classes, device=device)
    for class_weight in class_weights_arg.split(","):
        class_idx, weight = map(float, class_weight.split(":"))
        class_weights[int(class_idx)] = weight
    return class_weights


def initialize_model(config, in_channels=3, classes=20):
    if config.model == "DeepLabV3P+":
        decoder_atrous_rates = tuple(map(int, config.decoder_atrous_rates.split(",")))
        return smp.DeepLabV3Plus(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            encoder_output_stride=config.encoder_output_stride,
            decoder_channels=config.decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=in_channels,
            classes=classes,
            upsampling=config.upsampling,
        )
    elif config.model == "Unet":
        return smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            decoder_use_batchnorm=config.decoder_use_batchnorm == "True",
            decoder_attention_type=config.decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model}")


def initialize_loss_function(config, class_weights):
    """
    Initialize the loss function based on the config.
    """
    if config.loss == 'ce':
        return torch.nn.CrossEntropyLoss(weight=class_weights), "ce_loss"
    elif config.loss == 'dice':
        return DiceLoss(mode='multiclass', ignore_index=19), "dice_loss"
    elif config.loss == 'focal':
        return FocalLoss(mode='multiclass', ignore_index=19), "focal_loss"
    else:
        raise ValueError(f"Unsupported loss function: {config.loss}")


def kill_previous_processes():
    """
    Kill all Python processes running 'train.py' except the current one, to free GPU memory and display GPU memory usage.
    """
    try:
        # Get the PID of the current process
        current_pid = os.getpid()
        
        # Find all processes matching 'python train.py'
        output = subprocess.check_output("ps aux | grep 'python train.py' | grep -v grep", shell=True)
        for line in output.decode().splitlines():
            # Extract PID from the output
            pid = int(line.split()[1])
            if pid == current_pid:
                print(f"Skipping current process with PID: {current_pid}")
                continue
            print(f"Killing process with PID: {pid}")
            os.kill(pid, 9)  # Force kill the process
    except subprocess.CalledProcessError:
        print("No previous 'train.py' processes found.")
    
    # Display GPU memory usage using nvidia-smi
    try:
        memory_usage = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv", shell=True
        )
        print("Current GPU memory usage:")
        print(memory_usage.decode())
    except subprocess.CalledProcessError:
        print("Failed to retrieve GPU memory usage.")


if __name__ == '__main__':
    args = get_args()

    print_configuration(args)

    # WandB initialization
    wandb.init(
        project="cityscapes-runs",
        config=vars(args),
        name=f"{args.model}_{args.encoder}_BS_{args.batch_size_training}_Epochs_{args.epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    config = wandb.config

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment directories
    S_EXPERIMENT = wandb.run.name
    P_DIR_CKPT = os.path.join("./Workspace", S_EXPERIMENT, "Checkpoints")
    os.makedirs(P_DIR_CKPT, exist_ok=True)

    # Parse class weights
    class_weights = parse_class_weights(config.class_weights, num_classes=20, device=device)

    # Free up Memory and kill previous processes
    torch.cuda.empty_cache()
    kill_previous_processes()

    # Initialize model
    model = initialize_model(config)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)  # Automatically uses all available GPUs
        else:
            print("Using a single GPU")
    else:
        print("Using CPU")

    # Transfer model to device (either CPU or GPU)
    model.to(device)

    # Initialize loss function
    loss, loss_name = initialize_loss_function(config, class_weights)
    loss.__name__ = loss_name

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) if config.optimizer == "Adam" else torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Setup input normalization
    preprocess_input = get_preprocessing_fn(config.encoder, pretrained="imagenet")
    transform_crop = A.Compose([
        A.RandomCrop(config.patch_size, config.patch_size),
        # A.HorizontalFlip(p=0.25),
        A.Lambda(name="image_preprocessing", image=preprocess_input),
        A.Lambda(name="to_tensor", image=to_tensor)
    ])
    transform_full = A.Compose([
        A.Lambda(name="image_preprocessing", image=preprocess_input),
        A.Lambda(name="to_tensor", image=to_tensor)
    ])

    # Setup datasets
    dataset_training = DatasetCityscapesSemantic(
        root="./data", split="train", mode="fine", transform=transform_crop, device=device
    )
    dataset_validation = DatasetCityscapesSemantic(
        root="./data", split="val", mode="fine", transform=transform_full, device=device
    )
    dataset_test = DatasetCityscapesSemantic(
        root="./data", split="test", mode="fine", transform=transform_full, device=device
    )

    # Setup data loaders
    loader_training = DataLoader(
        dataset_training, batch_size=config.batch_size_training, shuffle=True, num_workers=config.num_workers
    )
    loader_validation = DataLoader(
        dataset_validation, batch_size=config.batch_size_validation, shuffle=False, num_workers=config.num_workers
    )

    # Setup training and validation
    epoch_training = Epoch(
        model, s_phase="training", loss=loss, optimizer=optimizer, device=device, verbose=True, writer=wandb
    )
    epoch_validation = Epoch(
        model, s_phase="validation", loss=loss, device=device, verbose=True, writer=wandb
    )

    # ======== TRAINING ======== #

    N_STEP_LOG = 1 # Logging Frequency
    max_score = 0
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch: {epoch} / {config.epochs} | LR = {round(scheduler.get_last_lr()[0], 8)}")

        # Training phase
        train_logs = epoch_training.run(loader_training, i_epoch=epoch)

        # Validation phase
        if epoch % N_STEP_LOG == 0:
            val_logs = epoch_validation.run(loader_validation, i_epoch=epoch)
            iou_score = val_logs.get("iou_score", 0)

            # Save best model
            if iou_score > max_score:
                max_score = iou_score
                torch.save(model, os.path.join(P_DIR_CKPT, f"best_model_epoch_{epoch:04d}.pth"))
                print(f"New best model saved with IoU: {max_score:.4f}")
                print("")

        # Adjust scheduler
        scheduler.step()

    # Finish WandB
    wandb.finish()

    # ======== TESTING ======== #
    # Uncomment to add testing logic
    # p_model_best = sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[-1]
    # print(f"Loading best model: {p_model_best}")
    # model.load_state_dict(torch.load(p_model_best))
    # test_epoch = Epoch(model, s_phase="test", loss=loss, p_dir_export=P_DIR_EXPORT, device=device, verbose=True)
    # test_epoch.run(loader_test)
