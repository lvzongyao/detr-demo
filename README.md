# DETR demo

A simple demo of how to use Facebook's DETR (DEtection TRansformer) object detector for inference. This repository contains a Jupyter Notebook that shows how to load a pretrained DETR model, run it on images, and visualize detections.

## Features
- Minimal, easy-to-follow Jupyter Notebook demonstrating DETR inference
- Example code for loading model weights and running detection on sample images
- Visualization of predicted bounding boxes and labels

## Prerequisites
- Python 3.8+
- PyTorch (install the version compatible with your CUDA / CPU): https://pytorch.org/
- torchvision
- Jupyter Notebook / JupyterLab

Optional (depending on notebook):
- pillow, matplotlib, numpy
- pycocotools (if COCO-format utilities are used)

If a requirements.txt is added to the repo, install with:
pip install -r requirements.txt

Or install the main dependencies manually:
pip install torch torchvision pillow matplotlib numpy jupyter

## Usage

1. Clone the repository:
   git clone https://github.com/lvzongyao/detr-demo.git
   cd detr-demo

2. Start Jupyter:
   jupyter notebook
   - or -
   jupyter lab

3. Open the notebook (e.g., DETR_demo.ipynb) and run the cells:
   - The notebook will show how to:
     - Load a pretrained DETR model
     - Preprocess images
     - Run inference
     - Display bounding boxes, class labels, and scores

4. Running in Google Colab:
   - Upload the notebook or open it directly from GitHub using Colab’s “Open with” feature.
   - Install dependencies in a Colab cell (install correct PyTorch wheel for Colab’s CUDA version).

## Model Weights
The notebook assumes access to DETR pretrained weights (e.g., the official weights provided by the DETR authors). If the notebook downloads weights automatically, ensure you have internet access. If you prefer local weights, update the notebook path to point to your local checkpoint.

Official DETR paper and model:
- DETR: End-to-End Object Detection with Transformers — https://arxiv.org/abs/2005.12872
- Official code and weights: https://github.com/facebookresearch/detr

## Example
A typical notebook flow:
- Load image(s)
- Preprocess to model input format
- model.eval(); outputs = model(inputs)
- Postprocess outputs to bounding boxes and class labels
- Visualize boxes with matplotlib or OpenCV

## Contributing
Contributions, suggestions, and bug reports are welcome. Please open an issue or submit a pull request with a clear description of changes.
