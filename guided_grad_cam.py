import albumentations as A
import cv2
import numpy as np
import torch
import os
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataset_cityscapes import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# ======== CONFIGURATION ======== #

S_NAME_CITY = "frankfurt"  # City for input image, e.g., "frankfurt"
IMAGE_NAME = "frankfurt_000001_029086_leftImg8bit.png"  # Specific image to visualize
S_NAME_ENCODER = "efficientnet-b4"
S_NAME_WEIGHTS = "imagenet"
P_DIR_MODEL = "./Workspace/DeepLabV3Plus_efficientnet-b4_BS_Train_8_Patch_512_Epochs_5/Checkpoints/best_model_epoch_0005.pth"
P_DIR_DATA = "./data"
P_DIR_OUTPUT = "./grad_cam"  # Directory to save Grad-CAM outputs
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = 6  # Target class index in Cityscapes (e.g., car, pedestrian, etc.)

# ======== SETUP ======== #

# Load model
model = torch.load(P_DIR_MODEL, map_location=S_DEVICE)
model.eval()
model = model.to(S_DEVICE)
for name, module in model.named_modules():
    print(name)  # Print all module names

# Input preprocessing
preprocess_input = get_preprocessing_fn(encoder_name=S_NAME_ENCODER, pretrained=S_NAME_WEIGHTS)
transform_full = A.Compose([
    A.Lambda(name="image_preprocessing", image=preprocess_input),
    A.Lambda(name="to_tensor", image=to_tensor),
])

# Load image
image_path = os.path.join(P_DIR_DATA, "leftImg8bit", "val", S_NAME_CITY, IMAGE_NAME)
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image_height, image_width, _ = original_image.shape


def preprocess_image(image):
    """
    Preprocess the image for the model.
    """
    image_tensor = transform_full(image=image)["image"]
    return torch.tensor(image_tensor).unsqueeze(0).float().to(S_DEVICE)


preprocessed_image = preprocess_image(original_image)


# ======== GRAD-CAM IMPLEMENTATION ======== #

def compute_grad_cam(model, image, target_class, target_layer):
    """
    Compute Grad-CAM for a specific layer.
    """
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the target layer
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break

    # Forward pass
    image.requires_grad = True
    output = model(image)
    target_score = output[:, target_class, :, :].mean()  # Target class score
    model.zero_grad()

    # Backward pass
    target_score.backward()

    # Extract gradients and activations
    gradient = gradients[0].cpu().detach().numpy()[0]  # Shape: [C, H, W]
    activation = activations[0].cpu().detach().numpy()[0]  # Shape: [C, H, W]

    # Grad-CAM computation
    weights = np.mean(gradient, axis=(1, 2))  # Global Average Pooling on gradients
    grad_cam = np.sum(weights[:, None, None] * activation, axis=0)  # Weighted sum
    grad_cam = np.maximum(grad_cam, 0)  # ReLU to keep positive contributions only

    # Normalize Grad-CAM
    grad_cam = cv2.resize(grad_cam, (image_width, image_height))
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

    return grad_cam


# ======== GRAD-CAM EVOLUTION ======== #

# Selected decoder layers for Grad-CAM evolution
decoder_layers = [
    "segmentation_head.2"
]

grad_cams = {}

# Compute Grad-CAM for each selected layer
for layer in decoder_layers:
    grad_cams[layer] = compute_grad_cam(model, preprocessed_image, TARGET_CLASS, layer)


# ======== VISUALIZATION ======== #

def visualize_grad_cam_evolution(original_image, grad_cams, save_path=None):
    """
    Visualize Grad-CAM evolution across selected layers.
    """
    plt.figure(figsize=(20, 10))
    num_layers = len(grad_cams)
    for idx, (layer_name, grad_cam) in enumerate(grad_cams.items()):
        plt.subplot(1, num_layers, idx + 1)
        plt.title(f"Layer: {layer_name}", fontsize=10)
        plt.imshow(original_image / 255.0, alpha=0.8)
        plt.imshow(grad_cam, cmap=cm.jet, alpha=0.5)
        plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# Save Grad-CAM evolution plot
os.makedirs(P_DIR_OUTPUT, exist_ok=True)
evolution_plot_path = os.path.join(P_DIR_OUTPUT, f"grad_cam_evolution_{IMAGE_NAME}")
visualize_grad_cam_evolution(original_image, grad_cams, save_path=evolution_plot_path)

print(f"Grad-CAM evolution plot saved to {evolution_plot_path}")