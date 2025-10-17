OCR Project: Text Detection and Recognition with YOLO and CRNN
This project implements an Optical Character Recognition (OCR) system using YOLO for text detection and a Convolutional Recurrent Neural Network (CRNN) for text recognition. The system processes images to detect text regions and recognize the text within those regions.
Project Structure

Dataset Preparation:

Extracts text annotations from an XML file and converts them into YOLO format for text detection.
Splits the dataset into train (80%), validation (15%), and test (5%) sets.
Organizes data into a YOLO-compatible structure (images/ and labels/ directories).
Creates a data.yaml configuration file for YOLO training.


Text Detection:

Uses a YOLO model (assumed to be YOLOv11, based on the project directory name) to detect text regions in images.
Trained model weights are stored at /content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt.


Text Recognition:

Employs a CRNN model with a ResNet152 backbone (in training) or ResNet101 (in inference) for recognizing text in cropped regions.
Trained CRNN weights are saved at /content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt.


Notebooks:

Untitled0.ipynb: Prepares the dataset, trains the CRNN model, and evaluates it.
Untitled1.ipynb: Loads the trained YOLO and CRNN models for inference (incomplete, lacks inference pipeline).



Prerequisites

Hardware: A GPU-enabled environment (e.g., Google Colab with T4 GPU) is recommended for training and inference.
Software:
Python 3.11
PyTorch 2.6 with CUDA support
Libraries: ultralytics, timm, torchvision, numpy, matplotlib, scikit-learn, pyyaml, Pillow



Installation

Clone the Repository (if applicable, or create a project directory):
mkdir ocr-yolo-crnn
cd ocr-yolo-crnn


Install Dependencies:
pip install ultralytics torch torchvision timm numpy matplotlib scikit-learn pyyaml Pillow


Set Up Google Drive (if using Google Colab):

Mount your Google Drive to access the dataset and models:from google.colab import drive
drive.mount('/content/drive')




Prepare the Dataset:

Place the dataset in /content/drive/MyDrive/OCR : Yolov11 vs CNN/SceneTrialTrain.
Ensure the dataset contains images and a words.xml file with text annotations.



Dataset

Location: /content/drive/MyDrive/OCR : Yolov11 vs CNN/SceneTrialTrain
Format:
Images: Stored in the dataset directory (e.g., JPG format).
Annotations: A words.xml file containing image paths, sizes, and bounding box coordinates with text labels.


YOLO Data:
Processed data is saved to /content/drive/MyDrive/OCR : Yolov11 vs CNN/yolo_data in the following structure:yolo_data/
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





Usage
1. Data Preparation
Run Untitled0.ipynb to:

Extract annotations from words.xml.
Convert bounding boxes to YOLO format.
Split and save the dataset into train, validation, and test sets.
Generate data.yaml for YOLO training.

2. Training the CRNN Model
In Untitled0.ipynb:

Define the STRDataset and CRNN classes.
Train the CRNN model for 100 epochs with the following hyperparameters:
Batch size: 64 (train), 128 (validation/test)
Optimizer: Adam (learning rate: 1e-5, weight decay: 1e-5)
Scheduler: StepLR (step size: 20, gamma: 0.5)
Loss: CTC loss


Save the trained model weights to /content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt.

3. Loading Models for Inference
In Untitled1.ipynb:

Load the YOLO model for text detection:from ultralytics import YOLO
yolo = YOLO('/content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt')


Load the CRNN model for text recognition:crnn = CRNN(vocab_size=len(chars) + 1, hidden_size=256, n_layers=3, dropout=0.2, unfreeze_layers=3).to(device)
load_crnn_weights(crnn, '/content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt', device)



Note: The inference pipeline is incomplete in Untitled1.ipynb. To perform end-to-end OCR:

Use the YOLO model to detect text regions in an image.
Crop the detected regions.
Preprocess the cropped images (grayscale, resize, normalize).
Pass the cropped images to the CRNN model for text recognition.
Decode the CRNN output using CTC decoding.

Example Inference Pipeline (To Be Implemented)
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

Model Details

YOLO Model:

Used for text detection.
Trained weights: /content/drive/MyDrive/OCR : Yolov11 vs CNN/best.pt
Class: Single class (text).


CRNN Model:

Backbone: ResNet152 (training) or ResNet101 (inference).
Recurrent Layer: Bidirectional GRU (256 hidden size, 3 layers).
Output: LogSoftmax over vocabulary (alphanumeric + space, hyphen, and blank character).
Loss: CTC loss for sequence alignment.
Training Results:
Validation loss: ~3.21
Test loss: ~3.44


Weights: /content/drive/MyDrive/OCR : Yolov11 vs CNN/ocr_crnn.pt



Warning: The CRNN model in Untitled1.ipynb uses ResNet101, which may not be fully compatible with the ResNet152 weights trained in Untitled0.ipynb. Ensure consistency in the backbone architecture or retrain the model.
Results

Training:

The CRNN model was trained for 100 epochs, achieving a validation loss of approximately 3.21 and a test loss of 3.44.
Training time per epoch: ~19 seconds on a Tesla T4 GPU.


Inference:

The inference pipeline is not fully implemented in the provided notebooks. Users must add code to integrate YOLO detections with CRNN recognition.



Troubleshooting

ResNet Mismatch: If loading CRNN weights fails due to ResNet101 vs. ResNet152, retrain the CRNN model using ResNet101 or modify Untitled0.ipynb to use ResNet101.
OutOfMemoryError: Reduce batch sizes (e.g., train_batch_size=32, test_batch_size=64) or clear CUDA cache (torch.cuda.empty_cache()).
Missing YOLO Training: The YOLO model training code is not provided. Ensure the best.pt weights are trained or available.

Future Improvements

Complete the inference pipeline in Untitled1.ipynb to integrate YOLO and CRNN.
Add CTC decoding for CRNN output to convert logits to readable text.
Evaluate text recognition accuracy (e.g., Character Error Rate) in addition to CTC loss.
Standardize the CRNN backbone (ResNet152 or ResNet101) across training and inference.
Document YOLO model training process or include training code.
