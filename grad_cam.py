import albumentations as A
import cv2
import numpy as np
import torch
import os
import warnings
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
P_DIR_MODEL = "./Workspace/DeepLabV3P+_efficientnet-b4_BS_60_Epochs_5_20241212_1747/Checkpoints/best_model_epoch_0005.pth"
P_DIR_DATA = "./data"
P_DIR_OUTPUT = "./grad_cam"  # Directory to save Grad-CAM outputs
S_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = 13  # Target class index in Cityscapes (e.g., car, pedestrian, etc.)

# Cityscapes categories
CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "pedestrian",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

CATEGORY_NAME = CITYSCAPES_CLASSES.get(TARGET_CLASS, "unknown")

# ======== SETUP ======== #

# Load model
try:
    model = torch.load(P_DIR_MODEL, map_location=S_DEVICE)
    model.eval()
    model = model.to(S_DEVICE)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {P_DIR_MODEL}")

model_to_inspect = model.module if isinstance(model, torch.nn.DataParallel) else model

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


# ======== GUIDED GRAD-CAM IMPLEMENTATION ======== #

def guided_grad_cam(model, image, target_class, target_layer):
    """
    Compute Guided Grad-CAM for the given model and input image.
    """
    # Handle DataParallel models
    model_to_inspect = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_inspect.eval()

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the target layer
    hook_registered = False
    for name, module in model_to_inspect.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)
            hook_registered = True
            break

    if not hook_registered:
        raise RuntimeError(f"Target layer '{target_layer}' not found in the model.")

    # Enable gradient computation for the input
    image.requires_grad = True

    # Forward pass
    output = model(image)
    target_score = output[:, target_class, :, :].mean()  # Target class score
    model.zero_grad()

    # Backward pass
    target_score.backward()

    # Ensure hooks captured values
    if not gradients:
        raise RuntimeError("Gradient hook failed to capture gradients.")
    if not activations:
        raise RuntimeError("Activation hook failed to capture activations.")

    # Extract gradients and activations
    gradient = gradients[0].cpu().detach().numpy()[0]  # Shape: [C, H, W]
    activation = activations[0].cpu().detach().numpy()[0]  # Shape: [C, H, W]

    # Grad-CAM computation
    weights = np.mean(gradient, axis=(1, 2))  # Global Average Pooling on gradients
    grad_cam = np.sum(weights[:, None, None] * activation, axis=0)  # Weighted sum
    grad_cam = np.maximum(grad_cam, 0)  # ReLU to keep positive contributions only

    # Normalize Grad-CAM
    grad_cam = cv2.resize(grad_cam, (image.shape[3], image.shape[2]))  # Match width, height
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

    # Guided Backpropagation
    guided_gradients = image.grad.abs().cpu().detach().numpy()[0]
    guided_grad_cam = guided_gradients.mean(axis=0) * grad_cam

    return grad_cam, guided_grad_cam



# Identify the target layer
# The target layer is specific to your model. Adjust this based on the printed model structure.
target_layer = "segmentation_head.0"  # Example: Choose the last convolutional layer

# Compute Guided Grad-CAM
grad_cam, guided_grad_cam = guided_grad_cam(model, preprocessed_image, TARGET_CLASS, target_layer)

# ======== SAVE IMAGES ======== #

# Create output directory if not exists
os.makedirs(P_DIR_OUTPUT, exist_ok=True)

# Save Original Image
original_path = os.path.join(P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_{CATEGORY_NAME}_original.png")
cv2.imwrite(original_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

# Save Grad-CAM Image
grad_cam_colormap = (cm.jet(grad_cam)[:, :, :3] * 255).astype(np.uint8)  # Convert to BGR format
grad_cam_overlay = cv2.addWeighted(original_image, 0.5, grad_cam_colormap, 0.5, 0)
grad_cam_path = os.path.join(P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_{CATEGORY_NAME}_grad_cam.png")
cv2.imwrite(grad_cam_path, cv2.cvtColor(grad_cam_overlay, cv2.COLOR_RGB2BGR))

# Save Guided Grad-CAM Image
guided_grad_cam_rescaled = (guided_grad_cam - guided_grad_cam.min()) / (guided_grad_cam.max() - guided_grad_cam.min())
guided_grad_cam_rescaled = (cm.hot(guided_grad_cam_rescaled)[:, :, :3] * 255).astype(np.uint8)
guided_grad_cam_path = os.path.join(P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_{CATEGORY_NAME}_guided_grad_cam.png")
cv2.imwrite(guided_grad_cam_path, guided_grad_cam_rescaled)

print(f"Images saved to {P_DIR_OUTPUT}:\n- {original_path}\n- {grad_cam_path}\n- {guided_grad_cam_path}")
