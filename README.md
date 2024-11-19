# Semantic Segmentation Cityscapes

This repository contains the implementation of a multi-class semantic segmentation pipeline for the popular Cityscapes [1] dataset, using PyTorch and the Segmentation Models Pytorch (SMP) [2] library. Models trained with this codebase generate predictions that can directly be submitted to the official [Benchmark Suite](https://www.cityscapes-dataset.com/benchmarks/).

|Image|Ground truth|Prediction|
|:-:|:-:|:-:|
|<img src="img/munster_000090_000019_leftImg8bit.png" width="266px"/>|<img src="img/munster_000090_000019_gtFine_color.png" width="266px"/>|<img src="img/munster_000090_000019_gtFine_color_prediction.png" width="266px"/>|

The code was developed by **Corentin Henry** and **Massimiliano Viola** for the **Image and Video Understanding** course held at TU Graz during the winter semester of 2022.

## Dependencies

A working installation of Python 3.7 or later is required. All the necessary libraries to run the scripts can then be installed with `pip install -r requirements.txt`. We recommend creating a virtual environment using a tool like [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Dataset

The Cityscapes Dataset focuses on semantic understanding of urban street scenes, with high-quality pixel-level annotations of 5000 frames for numerous cities and classes. The dataset is freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. More details and download are available at [www.cityscapes-dataset.com](https://www.cityscapes-dataset.com/).

## Train models

SMP easily allows us to train famous semantic segmentation architectures such as U-Net [3], DeepLabv3+ [4], and many others.

|U-Net|DeepLabV3+|
|:-:|:-:|
|<img src="img/UNet_architecture.png" width="400px" height="266px"/>|<img src="img/DeepLabV3+_architecture.png" width="400px" height="266px"/>|

In order to do so:
1. Edit `CUDA_VISIBLE_DEVICES` environment variable in `run.sh` to the GPU device IDs to use.
2. Modify training constants in `launch.py` to configure an experiment. In particular, choose between different encoders, weight initializations, architectures, augmentations and loss functions. See all available options in the [SMP documentation](https://smp.readthedocs.io/en/latest/). Set the max number of epochs and the crop patch size, updating the batch size and workers accordingly to your system.
3. Start from the terminal with `./run.sh` to train a model, save it, and run inference on the test images.

**Hardware Note:** all code has been tested and is set to run on a GPU with at least 16GB of VRAM, such as an NVIDIA® Tesla® P100 available for free on [Kaggle](https://www.kaggle.com/). Multi-GPU training was tested on two NVIDIA T4 GPUs available on Kaggle too. Modify the batch size or the patch size when using different hardware to fully take advantage of the graphic cards.

## Visualize models

By concatenating subsequent images of the test set with predictions overlayed, a time-lapse for a city can be generated. In order to do so:
1. In `visualization.py`, set the path to a model checkpoint, making sure to select the proper encoder and weight initialization.
2. Select a city, the number of images to predict, and the frame rate.
3. Run with `python3 ./visualization.py` to get the output video.

|DeepLabV3+ with EfficientNetB4 [5] backbone|
|:-:|
|<img src="img/berlin.gif" width="800px"/>|

## References

[1] Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding", https://www.cityscapes-dataset.com  
[2] Pavel Iakubovskii, "Segmentation Models Pytorch", https://github.com/qubvel/segmentation_models.pytorch  
[3] Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", https://arxiv.org/pdf/1505.04597.pdf  
[4] Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", https://arxiv.org/pdf/1802.02611.pdf  
[5] Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", https://arxiv.org/pdf/1905.11946.pdf  


## New One

### README for Semantic Segmentation Cityscapes

---

### Requirements

- **Python**: 3.11
- **pip**: ≥ 21.0
- **System Dependencies**:
  - `libgl1` (OpenCV support)
  - `libglib2.0-dev`
  - `libsm6`
  - `libxext6`
  - `libxrender1`

---

### Installation Guide

#### 1. **Set up Python Environment**
Ensure Python 3.11 is installed. Create a virtual environment:
```bash
python3.11 -m venv env
source env/bin/activate
```

#### 2. **Install Dependencies**
Upgrade pip and install project dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If system dependencies are missing, install them:
```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-dev libsm6 libxext6 libxrender1
```

#### 3. **Prepare Dataset**
Download the Cityscapes dataset and place it in the following structure:
```
./data/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

#### 4. **Start Training**
Adjust training configurations in `launch.py` if needed:
- `P_DIR_DATA = "./data"`
- Update `N_EPOCH_MAX`, `N_SIZE_PATCH`, etc., if required.

Run the training script:
```bash
bash run.sh
```

#### 5. **Test the Model**
The best model is saved in the checkpoints directory. Load it and execute the test phase separately:
```bash
python3 launch.py --test
```

#### 6. **Visualize Results**
Generate a video for predictions on the test set:
- Update `visualization.py` to set `P_DIR_MODEL` to the saved model path.
- Execute the script:
```bash
python3 visualization.py
```

#### 7. **TensorBoard Dashboard**
Visualize training and validation metrics using TensorBoard:
```bash
tensorboard --logdir=./Workspace/DeepLabV3+_EfficientNetB4_CE/Logs/
```
Open the provided URL in a browser.