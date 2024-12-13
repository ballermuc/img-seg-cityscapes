import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_cityscapes import DatasetCityscapesSemantic
import pandas as pd
import torch
import lookup_table as lut

# ======== CONFIGURATION ======== #

CITYSCAPES_ROOT = "./data"  # Update this to your dataset root
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

EDA_OUTPUT_DIR = "./eda_results"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

# ======== FUNCTIONS ======== #

def count_pixels_per_class(gt_mask, num_classes):
    """
    Count the number of pixels for each class in the ground truth mask.
    """
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for class_idx in range(num_classes):
        pixel_counts[class_idx] = np.sum(gt_mask == class_idx)
    return pixel_counts


def map_id_to_trainid(true_label, lut_id2trainid, device):
    """
    Map ground truth label IDs to train IDs using a lookup table.
    """
    true_label_tensor = torch.from_numpy(true_label).unsqueeze(0).unsqueeze(0).to(torch.uint8).to(device)
    lut_id2trainid = lut_id2trainid.to(device)
    return lut.lookup_nchw(true_label_tensor, lut_id2trainid).squeeze().cpu().numpy()


def plot_class_pixel_distribution(pixel_counts, class_names, output_path):
    """
    Plot the distribution of pixels per class as a bar plot sorted in descending order.
    """
    sorted_indices = np.argsort(pixel_counts)[::-1]
    sorted_pixel_counts = pixel_counts[sorted_indices]
    sorted_class_names = [class_names[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_class_names, y=sorted_pixel_counts, palette="viridis")
    plt.yscale("log")  # Log scale for scientific notation
    plt.xlabel("Classes")
    plt.ylabel("Number of Pixels (Log Scale)")
    plt.title("Pixel Distribution per Class")
    plt.xticks(rotation=45, ha="right")
    for i, count in enumerate(sorted_pixel_counts):
        plt.text(i, count, f"{count:.2e}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Bar plot saved to {output_path}")


# ======== EDA ANALYSIS ======== #

def perform_eda(data_dir, class_names, output_dir):
    """
    Perform EDA on the Cityscapes dataset to count pixels per class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_counts = np.zeros(len(class_names), dtype=np.int64)

    dataset = DatasetCityscapesSemantic(
        root=data_dir,
        split="val",
        mode="fine",
        device=device
    )

    # Iterate over the dataset and accumulate pixel counts
    for _, raw_labels, city, image_name in dataset:
        trainid_labels = map_id_to_trainid(raw_labels, dataset.th_i_lut_id2trainid, device)
        pixel_counts += count_pixels_per_class(trainid_labels, len(class_names))

    # Save results
    results_df = pd.DataFrame({
        "Class": class_names,
        "Pixel Count": pixel_counts
    }).sort_values(by="Pixel Count", ascending=False)
    csv_output_path = os.path.join(output_dir, "pixel_distribution.csv")
    results_df.to_csv(csv_output_path, index=False)
    print(f"Pixel distribution saved to {csv_output_path}")

    # Plot results
    barplot_output_path = os.path.join(output_dir, "pixel_distribution_barplot.png")
    plot_class_pixel_distribution(pixel_counts, class_names, barplot_output_path)


# ======== RUN ======== #

if __name__ == "__main__":
    perform_eda(
        data_dir=CITYSCAPES_ROOT,
        class_names=list(CITYSCAPES_CLASSES.values()),
        output_dir=EDA_OUTPUT_DIR
    )
