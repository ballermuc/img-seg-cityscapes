import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import albumentations as A
from dataset_cityscapes import DatasetCityscapesSemantic
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import lookup_table as lut
import cv2
import os


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def compute_grad_cam(model, image, target_class, target_layer):
    """
    Compute Grad-CAM for the given model and input image.
    """
    model_to_inspect = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_inspect.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    for name, module in model_to_inspect.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)
            break

    image.requires_grad = True
    output = model(image)
    target_score = output[:, target_class, :, :].mean()
    model.zero_grad()
    target_score.backward()

    if not gradients or not activations:
        raise RuntimeError("Hooks failed to capture gradients or activations.")

    gradient = gradients[0].cpu().detach().numpy()[0]
    activation = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(gradient, axis=(1, 2))
    grad_cam = np.sum(weights[:, None, None] * activation, axis=0)
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (image.shape[3], image.shape[2]))
    return grad_cam


def compute_confusion_matrix_all_images(model, dataloader, lut_id2trainid, target_layer, class_names, device):
    """
    Compute a confusion matrix for Grad-CAM activations aggregated over the first 50 images.
    """
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    for i, (images, raw_labels, _, _) in enumerate(tqdm(dataloader, desc="Processing validation images")):
        if i >= 50:  # Limit to the first 50 images
            break
        images = images.to(device)
        true_labels = np.stack([
            map_id_to_trainid(label.cpu().numpy(), lut_id2trainid, device) for label in raw_labels
        ])

        for target_class_idx in range(num_classes):
            grad_cam = compute_grad_cam(model, images, target_class_idx, target_layer)
            grad_cam_normalized = np.clip(grad_cam, a_min=0, a_max=None)

            for gt_class_idx in range(num_classes):
                mask = (true_labels == gt_class_idx)
                confusion_matrix[target_class_idx, gt_class_idx] += np.sum(grad_cam_normalized * mask)

    return confusion_matrix


def plot_confusion_matrix_real_sum_row_normalized(confusion_matrix, class_names, output_path):
    """
    Plot and save a confusion matrix where coloring is normalized per row.

    Args:
        confusion_matrix (np.ndarray): Raw confusion matrix.
        class_names (list): List of class names.
        output_path (str): Path to save the confusion matrix plot.
    """
    # Convert confusion matrix to a DataFrame for normalization
    confusion_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    
    # Normalize each row
    row_normalized_df = confusion_df.div(confusion_df.sum(axis=1), axis=0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        row_normalized_df,
        annot=True,
        fmt=".2f",  # Display normalized values with two decimals
        cmap="YlGnBu",
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10},
    )
    plt.title("Row-Normalized Aggregated Grad-CAM Confusion Matrix")
    plt.xlabel("Ground Truth Classes")
    plt.ylabel("Target Classes")
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Row-normalized confusion matrix saved to {output_path}")



def map_id_to_trainid(true_label, lut_id2trainid, device):
    """
    Map ground truth label IDs to train IDs using a lookup table.
    """
    true_label_tensor = torch.from_numpy(true_label).unsqueeze(0).unsqueeze(0).to(torch.uint8).to(device)
    lut_id2trainid = lut_id2trainid.to(device)
    return lut.lookup_nchw(true_label_tensor, lut_id2trainid).squeeze().cpu().numpy()


# ======== Configuration ======== #
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

P_DIR_MODEL = "./Workspace/DeepLabV3P+_efficientnet-b4_BS_60_Epochs_5_20241212_1747/Checkpoints/best_model_epoch_0005.pth"
P_DIR_OUTPUT = "./grad_cam"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load(P_DIR_MODEL, map_location=device)
model.eval()

dataset_validation = DatasetCityscapesSemantic(
    root="./data",
    split="val",
    mode="fine",
    device=device,
    transform=A.Compose([
        A.Lambda(name="to_tensor", image=to_tensor),
    ]),
)
dataloader = DataLoader(dataset_validation, batch_size=1, shuffle=False)
target_layer = "segmentation_head.0"
class_names = list(CITYSCAPES_CLASSES.values())

# ======== Compute Confusion Matrix ======== #
confusion_matrix = compute_confusion_matrix_all_images(
    model, dataloader, dataset_validation.th_i_lut_id2trainid, target_layer, class_names, device
)

# ======== Save and Plot Confusion Matrix ======== #
# Extract model details from P_DIR_MODEL
model_name = os.path.basename(os.path.dirname(P_DIR_MODEL))  # e.g., "DeepLabV3P+_efficientnet-b4_BS_60_Epochs_5_20241212_1747"
epoch_name = os.path.splitext(os.path.basename(P_DIR_MODEL))[0].split('_')[-1]  # e.g., "0005"
output_filename = f"{model_name}_epoch_{epoch_name}_confusion.png"

# Define the folder structure
P_DIR_ACTIVATIONS = os.path.join(P_DIR_OUTPUT, "activation_confusion_matrix")
os.makedirs(P_DIR_ACTIVATIONS, exist_ok=True)  # Create directory if it doesn't exist

# Set full path for the output file
confusion_matrix_output_path = os.path.join(P_DIR_ACTIVATIONS, output_filename)
plot_confusion_matrix_real_sum_row_normalized(confusion_matrix, class_names, confusion_matrix_output_path)
print(f"Row-normalized confusion matrix saved to {confusion_matrix_output_path}")
