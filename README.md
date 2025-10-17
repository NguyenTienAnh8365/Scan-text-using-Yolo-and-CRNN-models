# OCR Project: Text Detection and Recognition with YOLO and CRNN

This project implements an Optical Character Recognition (OCR) system using YOLO for text detection and a Convolutional Recurrent Neural Network (CRNN) for text recognition. The system processes images to detect text regions and recognize the text within those regions.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
This OCR system combines:
- **YOLO**: Detects text regions in images (assumed to be YOLOv11, based on directory naming).
- **CRNN**: Recognizes text in cropped regions using a ResNet backbone (ResNet152 for training, ResNet101 for inference) and bidirectional GRU.

The project includes dataset preparation, model training, and an incomplete inference pipeline.

## Project Structure
- **Dataset Preparation**:
  - Extracts text annotations from an XML file and converts them to YOLO format.
  - Splits dataset into train (80%), validation (15%), and test (5%) sets.
  - Organizes data into YOLO-compatible structure (`images/` and `labels/` directories).
  - Creates a `data.yaml` configuration file for YOLO training.

- **Text Detection**:
  - Uses a YOLO model for text detection.
  - Trained weights: `/content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt`.

- **Text Recognition**:
  - Employs a CRNN model for text recognition in cropped regions.
  - Trained weights: `/content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt`.

- **Notebooks**:
  - `Untitled0.ipynb`: Prepares dataset, trains CRNN model, and evaluates it.
  - `Untitled1.ipynb`: Loads YOLO and CRNN models for inference (incomplete).

## Prerequisites
### Hardware
- GPU-enabled environment (e.g., Google Colab with T4 GPU) recommended for training and inference.

### Software
- Python 3.11
- PyTorch 2.6 with CUDA support
- Libraries: `ultralytics`, `timm`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `pyyaml`, `Pillow`

## Installation
1. Clone the repository or create a project directory:
   ```bash
   mkdir ocr-yolo-crnn
   cd ocr-yolo-crnn
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics torch torchvision timm numpy matplotlib scikit-learn pyyaml Pillow
   ```

3. Set up Google Drive (if using Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Place the dataset in `/content/drive/MyDrive/OCR : Yolov11 vs CNN/SceneTrialTrain` with images and a `words.xml` file containing text annotations.

## Dataset
- **Location**: `/content/drive/MyDrive/OCR : Yolov11 vs CNN/SceneTrialTrain`
- **Format**:
  - **Images**: JPG format, stored in the dataset directory.
  - **Annotations**: `words.xml` file with image paths, sizes, and bounding box coordinates with text labels.
- **YOLO Data**: Processed data saved to `/content/drive/MyDrive/OCR : Yolov11 vs CNN/yolo_data`:
  ```
  yolo_data/
  ├── train/
  │   ├── images/
  │   ├── labels/
  ├── val/
  │   ├── images/
  │   ├── labels/
  ├── test/
  │   ├── images/
  │   ├── labels/
  ├── data.yaml
  ```

## Usage
### 1. Data Preparation
Run `Untitled0.ipynb` to:
- Extract annotations from `words.xml`.
- Convert bounding boxes to YOLO format.
- Split dataset into train, validation, and test sets.
- Generate `data.yaml` for YOLO training.

### 2. Training the CRNN Model
In `Untitled0.ipynb`:
- Define `STRDataset` and `CRNN` classes.
- Train the CRNN model for 100 epochs with:
  - **Batch size**: 64 (train), 128 (validation/test)
  - **Optimizer**: Adam (learning rate: 1e-5, weight decay: 1e-5)
  - **Scheduler**: StepLR (step size: 20, gamma: 0.5)
  - **Loss**: CTC loss
- Save weights to `/content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt`.

### 3. Loading Models for Inference
In `Untitled1.ipynb`:
- Load YOLO model:
  ```python
  from ultralytics import YOLO
  yolo = YOLO('/content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt')
  ```
- Load CRNN model:
  ```python
  crnn = CRNN(vocab_size=len(chars) + 1, hidden_size=256, n_layers=3, dropout=0.2, unfreeze_layers=3).to(device)
  load_crnn_weights(crnn, '/content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt', device)
  ```

### 4. Example Inference Pipeline (To Be Implemented)
```python
from PIL import Image
import torchvision.transforms as transforms

# Load image
image_path = 'path/to/image.jpg'
image = Image.open(image_path).convert('RGB')

# Text detection with YOLO
results = yolo(image)
boxes = results[0].boxes.xyxy  # Get bounding boxes

# Preprocess for CRNN
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Recognize text in each bounding box
crnn.eval()
with torch.no_grad():
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image.crop((x1, y1, x2, y2))
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        output = crnn(input_tensor)
        # Decode output using CTC (e.g., torch.argmax or CTC decoder)
        # Process and print the recognized text
```

> **Note**: The inference pipeline in `Untitled1.ipynb` is incomplete. Implement the above pipeline to integrate YOLO detections with CRNN recognition.

## Model Details
### YOLO Model
- **Purpose**: Text detection
- **Trained weights**: `/content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt`
- **Class**: Single class (text)

### CRNN Model
- **Backbone**: ResNet152 (training) or ResNet101 (inference)
- **Recurrent Layer**: Bidirectional GRU (256 hidden size, 3 layers)
- **Output**: LogSoftmax over vocabulary (alphanumeric + space, hyphen, and blank character)
- **Loss**: CTC loss for sequence alignment
- **Training Results**:
  - Validation loss: ~3.21
  - Test loss: ~3.44
- **Weights**: `/content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt`

> **Warning**: The CRNN model in `Untitled1.ipynb` uses ResNet101, which may not be compatible with ResNet152 weights from `Untitled0.ipynb`. Ensure consistency in the backbone or retrain.

## Results
- **Training**:
  - CRNN trained for 100 epochs.
  - Validation loss: ~3.21
  - Test loss: ~3.44
  - Training time: ~19 seconds per epoch on a Tesla T4 GPU.
- **Inference**:
  - Inference pipeline is incomplete. Users must implement the integration of YOLO and CRNN.

## Troubleshooting
- **ResNet Mismatch**: If loading CRNN weights fails due to ResNet101 vs. ResNet152, retrain the CRNN model with ResNet101 or modify `Untitled0.ipynb` to use ResNet101.
- **OutOfMemoryError**: Reduce batch sizes (e.g., `train_batch_size=32`, `test_batch_size=64`) or clear CUDA cache (`torch.cuda.empty_cache()`).
- **Missing YOLO Training**: YOLO training code is not provided. Ensure `best.pt` weights are trained or available.

## Future Improvements
- Complete the inference pipeline in `Untitled1.ipynb`.
- Add CTC decoding for CRNN output to convert logits to readable text.
- Evaluate text recognition accuracy (e.g., Character Error Rate).
- Standardize CRNN backbone (ResNet152 or ResNet101) across training and inference.
- Document YOLO model training process or include training code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.