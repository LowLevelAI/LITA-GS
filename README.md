#  [CVPR 2025] LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors [[Paper]](https://arxiv.org/pdf/2504.00219)

<h4 align="center"> Han Zhou<sup></sup>, Wei Dong<sup>&dagger;</sup>, Jun Chen<sup></sup></center>
<h4 align="center"> McMaster University, <sup>&dagger;</sup>Corresponding Author</center></center>
  
### Introduction
This repository represents the official implementation of our CVPR 2025 paper titled **LITA-GS: Illumination-Agnostic Novel View Synthesis via Reference-Free 3D Gaussian Splatting and Physical Priors**. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you for your interest. 

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

## 📢 News
**2025-06-12** We strat to update this repo from today, and we plan to make it complete within one week!

### Overall Framework
![teaser](images/framework.png)


## 🛠️ Setup

The code was tested on:

- RTX 5090, Python 3.9, CUDA 12.8, PyTorch 2.8 + cu12.8.

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/LowLevelAI/LITA-GS.git
cd LITA-GS
```

### 💻 Dependencies

- **Create the Conda environment:** 

    ```bash
    conda create -n litags python=3.9
    conda activate litags
    ```
- **Then install dependencies:**
  - Install Pytorch

  ```bash
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
  - Set Cudatoolkit to 12.8
  ```bash
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
  ```
  - Install dependencies
  ```bash
  pip install trimesh tqdm mmcv==1.6.0 scipy scikit-image
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/simple-knn
  ```
