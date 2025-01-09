# Advanced Machine Learning Project: Image Segmentation on the Cityscapes Dataset

**Course:** AML4KCS2024 - Advanced Machine Learning at ITU  
**Project Title:** Impact of Loss Functions on Class Imbalance in Semantic Segmentation for Autonomous Driving
**Contacts:**  
- Julius Freese <julfr@itu.dk>  
- Fynn Louis Schröder <fysc@itu.dk>  



---

## How to Run

### 1. Clone the Repository
```bash
git clone --depth 1 https://github.com/ballermuc/img-seg-cityscapes.git
cd /img-seg-cityscapes
```

### 2. Set Up the Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Cityscapes Data
- Download the dataset from [Cityscapes Dataset](https://www.cityscapes-dataset.com/).  
- Place the downloaded data in the `data/` folder within the repository.

### 5. Run the Training Script
```bash
bash run.sh
```

### 6. Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

---


## Sources
This project includes original code as well as adapted components from the following sources:
- **[semantic-segmentation-cityscapes](https://github.com/massimilianoviola/semantic-segmentation-cityscapes):** Parts of the code used in this project, such as data processing pipelines and training utilities, were adapted from the repository by Corentin Henry and Massimiliano Viola. See full citation below:
@misc{henry2022github, author = {Henry, Corentin and Viola, Massimiliano}, title = {semantic-segmentation-cityscapes}, year = {2022}, howpublished = {\url{https://github.com/massimilianoviola/semantic-segmentation-cityscapes}}, note = {GitHub repository} }
- **[Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch):** Used for model implementations and pre-trained weights.

## License
The adaptations from the **semantic-segmentation-cityscapes** repository are used under the terms of the MIT License provided in that repository.

## Acknowledgments
We acknowledge the contributions of the authors of the original repositories for their foundational work, which facilitated the development of this project.


