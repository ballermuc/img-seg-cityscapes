import albumentations as A
import cv2
import numpy as np
import seaborn as sns  
import pandas as pd
import torch
import os
import warnings
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataset_cityscapes import *
from dataset_cityscapes import DatasetCityscapesSemantic
import lookup_table as lut



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
TARGET_CLASS = 6  # Target class index in Cityscapes (e.g., car, pedestrian, etc.)

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

def compute_grad_cam(model, image, target_class, target_layer):
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
grad_cam, guided_grad_cam = compute_grad_cam(model, preprocessed_image, TARGET_CLASS, target_layer)

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



# ======== Compute Activation-Weighted Mask ======== #
def save_activation_barplot_with_gt_distribution(grad_cam, true_label, class_names, output_path, target_class_name):
    """
    Create and save a bar plot of ground truth class distribution for Grad-CAM activations of the target class.
    """
    grad_cam_normalized = np.clip(grad_cam, a_min=0, a_max=None)

    activation_sums = [
        np.sum(grad_cam_normalized * (true_label == class_idx))
        for class_idx in range(len(class_names))
    ]

    sorted_indices = np.argsort(activation_sums)[::-1]
    sorted_sums = [activation_sums[i] for i in sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_class_names, sorted_sums, color="skyblue")
    plt.xlabel("Ground Truth Classes")
    plt.ylabel("Activation Sum")
    plt.title(f"Activation Sum Distribution for '{target_class_name}' Activations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_ground_truth_mask(city, image_name, data_dir):
    mask_name = image_name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    mask_path = os.path.join(data_dir, "gtFine", "val", city, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to load ground truth mask: {mask_path}")
    return mask


def map_id_to_trainid(true_label, lut_id2trainid, device):
    """
    Map ground truth label IDs to train IDs using a lookup table.
    Ensure tensor and LUT are on the same device.
    """
    true_label_tensor = torch.from_numpy(true_label).unsqueeze(0).unsqueeze(0).to(torch.uint8).to(device)
    lut_id2trainid = lut_id2trainid.to(device)
    return lut.lookup_nchw(true_label_tensor, lut_id2trainid).squeeze().cpu().numpy()


# ======== Initialize Dataset and Load Ground Truth ======== #
from dataset_cityscapes import DatasetCityscapesSemantic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = DatasetCityscapesSemantic(
    device=device,
    root="./data",  # Update with your actual data directory
    split="val",
    mode="fine",
)

true_label_raw = load_ground_truth_mask(S_NAME_CITY, IMAGE_NAME, P_DIR_DATA)
true_label = map_id_to_trainid(true_label_raw, dataset.th_i_lut_id2trainid, device)

# ======== Compute Grad-CAM ======== #
grad_cam_output, _ = compute_grad_cam(model, preprocessed_image, TARGET_CLASS, target_layer)

# ======== Apply Grad-CAM Mask and Compute Bar Plot ======== #
barplot_output_path = os.path.join(
    P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_{CATEGORY_NAME}_gt_distribution_barplot.png"
)
save_activation_barplot_with_gt_distribution(
    grad_cam_output,
    true_label,
    list(CITYSCAPES_CLASSES.values()),
    barplot_output_path,
    CATEGORY_NAME
)


####################################
def save_combined_activation_barplot(grad_cam_dict, true_label, class_names, output_path):
    """
    Create and save a combined bar plot of ground truth class distribution
    for Grad-CAM activations of all target classes, with distinct separation for each target class.
    """
    activation_sums_dict = {}

    for target_class_name, grad_cam in grad_cam_dict.items():
        grad_cam_normalized = np.clip(grad_cam, a_min=0, a_max=None)
        activation_sums = [
            np.sum(grad_cam_normalized * (true_label == class_idx))
            for class_idx in range(len(class_names))
        ]
        activation_sums_dict[target_class_name] = activation_sums

    # Create a combined bar plot with separated sections for each target class
    x_labels = class_names
    num_classes = len(class_names)
    x_spacing = num_classes + 3  # Add spacing between groups
    x_positions = np.arange(0, len(activation_sums_dict) * x_spacing, x_spacing)  # Group positions

    plt.figure(figsize=(20, 10))
    bar_width = 0.8  # Width of each bar
    colors = plt.cm.tab20.colors  # Use a colormap for consistent coloring

    for idx, (target_class_name, activation_sums) in enumerate(activation_sums_dict.items()):
        # Calculate positions for bars of this group
        x = x_positions[idx] + np.arange(num_classes)

        plt.bar(
            x,
            activation_sums,
            bar_width,
            label=f"Target Class: {target_class_name}",
            color=colors[idx % len(colors)],
        )
        # Add a vertical separator between groups for clarity
        if idx > 0:
            plt.axvline(x=x_positions[idx] - 1.5, color='gray', linestyle='--', linewidth=1.2)

    plt.xlabel("Ground Truth Classes")
    plt.ylabel("Activation Sum")
    plt.title("Activation Sum Distribution Across All Target Classes")
    plt.xticks(
        np.concatenate([x_positions[i] + np.arange(num_classes) for i in range(len(activation_sums_dict))]),
        [class_name for _ in range(len(activation_sums_dict)) for class_name in class_names],
        rotation=45,
        ha="right"
    )
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()
    print(f"Combined bar plot saved at {output_path}")


# ======== Compute Grad-CAM for All Target Classes ======== #
grad_cam_dict = {}

for target_class, target_class_name in CITYSCAPES_CLASSES.items():
    grad_cam_output, _ = compute_grad_cam(model, preprocessed_image, target_class, target_layer)
    grad_cam_dict[target_class_name] = grad_cam_output

# ======== Apply Grad-CAM Mask and Compute Combined Bar Plot ======== #
combined_barplot_output_path = os.path.join(
    P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_combined_gt_distribution_barplot.png"
)

save_combined_activation_barplot(
    grad_cam_dict,
    true_label,
    list(CITYSCAPES_CLASSES.values()),
    combined_barplot_output_path
)




################## CONFUSION ############
def compute_confusion_matrix_real_sum(grad_cam_dict, true_label, class_names):
    """
    Compute a confusion matrix using real activation sums from Grad-CAM activations.

    Args:
        grad_cam_dict (dict): Grad-CAM activations for all target classes.
        true_label (np.ndarray): Ground truth labels (mapped to train IDs).
        class_names (list): List of class names.

    Returns:
        pd.DataFrame: Confusion matrix as a DataFrame.
    """
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    for target_class_idx, (target_class_name, grad_cam) in enumerate(grad_cam_dict.items()):
        grad_cam_normalized = np.clip(grad_cam, a_min=0, a_max=None)

        for gt_class_idx in range(num_classes):
            # Compute total activation for each ground truth class
            class_activation_sum = np.sum(grad_cam_normalized * (true_label == gt_class_idx))
            confusion_matrix[target_class_idx, gt_class_idx] = class_activation_sum

    # Convert to DataFrame for better handling with Seaborn
    confusion_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    return confusion_df


def plot_confusion_matrix_real_sum_row_normalized(confusion_df, output_path):
    """
    Plot and save a confusion matrix where coloring is normalized per row.

    Args:
        confusion_df (pd.DataFrame): Confusion matrix as a DataFrame.
        output_path (str): Path to save the confusion matrix plot.
    """
    # Normalize each row
    row_normalized_df = confusion_df.div(confusion_df.sum(axis=1), axis=0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        row_normalized_df,
        annot=True,
        fmt=".2f",  # Show normalized values with two decimals
        cmap="YlGnBu",  # Use a perceptually uniform colormap
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10},
    )
    plt.title("Row-Normalized Grad-CAM Confusion Matrix")
    plt.xlabel("Ground Truth Classes")
    plt.ylabel("Target Classes")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Row-normalized confusion matrix saved at {output_path}")


# ======== Compute Grad-CAM Confusion Matrix ======== #
confusion_df_real_sum = compute_confusion_matrix_real_sum(grad_cam_dict, true_label, list(CITYSCAPES_CLASSES.values()))

# ======== Plot Confusion Matrix ======== #
confusion_matrix_real_sum_output_path = os.path.join(
    P_DIR_OUTPUT, f"{IMAGE_NAME.split('.')[0]}_grad_cam_confusion_matrix_real_sum.png"
)
plot_confusion_matrix_real_sum_row_normalized(confusion_df_real_sum, confusion_matrix_real_sum_output_path)




