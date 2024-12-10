import albumentations as A
import cv2
import datetime
from glob import glob
import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_cityscapes import *
from epoch import *
import wandb


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# p_* stands for "path"
# s_* stands for "string" (i.e. all other strings than paths)
# n_* stands for n-dim size (0-dim: number of objects, 1-dim+: shape)

P_DIR_DATA = "./data"
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "DeepLabV3P+"
S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
N_EPOCH_MAX = 20
N_SIZE_BATCH_TRAINING = 8  # training batch size
N_SIZE_BATCH_VALIDATION = 4  # validation batch size
N_SIZE_BATCH_TEST = 1  # test batch size
N_SIZE_PATCH = 512  # patch size for random crop
N_STEP_LOG = 1  # evaluate on validation set and save model every N iterations
N_WORKERS = 16  # to be adapted for each system
# other notations:
# l_* stands for "list"
# i_* stands for an index, e.g.: for i_object, object in enumerate(l_object):
# d_* stands for "dict"
# k_* stands for "key" (of a dictionary item)

S_EXPERIMENT = (
    f"{MODEL_NAME}_"
    f"{S_NAME_ENCODER}_"
    f"BS_Train_{N_SIZE_BATCH_TRAINING}_"
    f"Patch_{N_SIZE_PATCH}_"
    f"Epochs_{N_EPOCH_MAX}"
)
P_DIR_CKPT = os.path.join("./Workspace", S_EXPERIMENT, "Checkpoints")
P_DIR_LOGS = os.path.join("./Workspace", S_EXPERIMENT, "Logs")
P_DIR_EXPORT = os.path.join("./Workspace", S_EXPERIMENT, "Export")

# Initialize wandb
wandb.init(
    project="img-seg-cityscapes",
    name=S_EXPERIMENT,
    config={
        "epochs": N_EPOCH_MAX,
        "batch_size_training": N_SIZE_BATCH_TRAINING,
        "batch_size_validation": N_SIZE_BATCH_VALIDATION,
        "batch_size_test": N_SIZE_BATCH_TEST,
        "learning_rate": 5e-4,
        "model": MODEL_NAME,
        "encoder": S_NAME_ENCODER,
        "optimizer": "Adam",
    }
)

wandb.config.update({
    "patch_size": N_SIZE_PATCH,
    "num_workers": N_WORKERS
})


# ======== SETUP ======== #

class CustomDeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(self, *args, dropout_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        # Access the ASPP module and modify its project layer
        self.decoder.aspp[0].project = nn.Sequential(
            self.decoder.aspp[0].project,
            nn.Dropout(dropout_rate)
        )


# # Example usage
model = CustomDeepLabV3Plus(
    encoder_name= S_NAME_ENCODER,
    encoder_weights=S_NAME_WEIGHTS,
    in_channels=3,
    classes=20,
    dropout_rate=0.5
)


# # setup model
# model = smp.DeepLabV3Plus(
#     encoder_name = S_NAME_ENCODER,
#     encoder_weights = S_NAME_WEIGHTS,
#     in_channels = 3,
#     classes = 20,
# )

# print(model)


# U-Net model
#model = smp.Unet(
#    encoder_name = S_NAME_ENCODER,
#    encoder_weights = S_NAME_WEIGHTS,
#    in_channels = 3,
#    classes = 20,
#)
# To enable multi-GPU training. Set device_ids accordingly.
#model = torch.nn.DataParallel(model, device_ids=[0, 1])

# Watch the model to log parameters (weights, biases) and gradients
wandb.watch(model, log="all", log_freq=100)

# setup input normalization
preprocess_input = get_preprocessing_fn(
    encoder_name = S_NAME_ENCODER,
    pretrained = S_NAME_WEIGHTS,
)
# setup input augmentations (for cropped and full images)
transform_crop = A.Compose([
    A.RandomCrop(N_SIZE_PATCH, N_SIZE_PATCH),
#    A.HorizontalFlip(p=0.25),
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
transform_full = A.Compose([
    A.Lambda(name = "image_preprocessing", image = preprocess_input),
    A.Lambda(name = "to_tensor", image = to_tensor),
])
# setup datasets
dataset_training = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "train",
    mode = "fine",
    transform = transform_crop,
    device = S_DEVICE,
)
dataset_validation = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "val",
    mode = "fine",
    transform = transform_full,
    device = S_DEVICE,
)
dataset_test = DatasetCityscapesSemantic(
    root = P_DIR_DATA,
    split = "test",
    mode = "fine",
    transform = transform_full,
    device = S_DEVICE,
)
# setup data loaders
loader_training = DataLoader(
    dataset_training,
    batch_size = N_SIZE_BATCH_TRAINING,
    shuffle = True,
    num_workers = N_WORKERS,
)
loader_validation = DataLoader(
    dataset_validation,
    batch_size = N_SIZE_BATCH_VALIDATION,
    shuffle = False,
    num_workers = N_WORKERS,
)
loader_test = DataLoader(
    dataset_test,
    batch_size = N_SIZE_BATCH_TEST,
    shuffle = False,
    num_workers = N_WORKERS,
)
# setup loss
loss = torch.nn.CrossEntropyLoss()
loss.__name__ = "ce_loss"

# To use SMP losses
#loss = smp.losses.DiceLoss(mode="multiclass")
#loss.__name__ = "dice_loss"
# setup optimizer
optimizer = torch.optim.Adam([
    dict(params = model.parameters(), lr = 5e-4),
])
# setup learning rate scheduler
# (here cosine decay that reaches 1e-6 after N_EPOCH_MAX)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer = optimizer,
    T_max = N_EPOCH_MAX,
    eta_min = 1e-6,
)


# ======== TRAINING ======== #

# initialize training instance
epoch_training = Epoch(
    model,
    s_phase = "training",
    loss = loss,
    optimizer = optimizer,
    device = S_DEVICE,
    verbose = True,
    writer = wandb,
)
# initialize validation instance
epoch_validation = Epoch(
    model,
    s_phase = "validation",
    loss = loss,
    device = S_DEVICE,
    verbose = True,
    writer = wandb,
)
# start training phase
os.makedirs(P_DIR_CKPT, exist_ok = True)
max_score = 0
# iterate over epochs
for i in range(1, N_EPOCH_MAX + 1):
    print(f"Epoch: {i} | LR = {round(scheduler.get_last_lr()[0], 8)}")
    d_log_training = epoch_training.run(loader_training, i_epoch = i)
    iou_score = round(d_log_training["iou_score"] * 100, 2)
    print(f"IoU = {iou_score}%")
    print()
    # log validation performance
    if i % N_STEP_LOG == 0:
        d_log_validation = epoch_validation.run(loader_validation, i_epoch = i)
        iou_score = round(d_log_validation["iou_score"] * 100, 2)
        print(f"IoU = {iou_score}%")
        # save model if better than previous best
        if max_score < iou_score:
            max_score = iou_score
            wandb.unwatch(model)
            torch.save(model, os.path.join(P_DIR_CKPT, f"best_model_epoch_{i:0>4}.pth"))
            wandb.watch(model, log="all", log_freq=100)
            print("Model saved!")
            wandb.log({"Best IoU": max_score, "Epoch": i})
        print()
    scheduler.step()

# Finish WandB session
wandb.finish()

# # ======== TEST ======== #
# print("\n==== TEST PHASE====\n")
# # create export directory
# os.makedirs(P_DIR_EXPORT, exist_ok = True)
# # load best model
# p_model_best = sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[-1]
# print(f"Loading following model: {p_model_best}")
# model = torch.load(p_model_best)
# # initialize test instance
# test_epoch = Epoch(
#     model,
#     s_phase = "test",
#     loss = loss,
#     p_dir_export = P_DIR_EXPORT,
#     device = S_DEVICE,
#     verbose = True,
# )
# test_epoch.run(loader_test)
# # remove intermediate checkpoints
# for model_checkpoint in sorted(glob(os.path.join(P_DIR_CKPT, "*.pth")))[:-1]:
#     os.remove(model_checkpoint)
