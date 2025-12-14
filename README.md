# 🐶 Cat vs. Dog Image Classifier 🐱

A Production-Ready end-to-end Machine Learning pipeline that classifies images of cats and dogs.
Built with **PyTorch** for the CNN model and **Streamlit** for the web interface. 

This project demonstrates a complete workflow:
* **Data Pipeline:** Automated processing of raw data into training/validation sets.
* **Custom CNN:** A lightweight Convolutional Neural Network built from scratch.
* **Deployment:** A user-friendly web app for real-time inference.

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Deep Learning:** PyTorch (with CUDA support for GPU acceleration)
* **Web Framework:** Streamlit
* **Data Processing:** Torchvision, PIL

### Prerequisites
* NVIDIA GPU (Recommended) with CUDA drivers (Tested on RTX 3050 with CUDA 13.1)
* Python 3.10 or 3.11 (Python 3.1x > 3.11.9 is currently unsupported by PyTorch)

## 📂 Project Structure

```text
cat_dog_project/
├── data/
│   ├── PetImages/      # Raw Data
│   ├── train/          # Processed Training Data
│   └── val/            # Processed Validation Data
├── models/
│   └── cat_dog_model.pth  # Saved trained model
├── src/
│   ├── app.py          # Streamlit Web Application
│   ├── data_setup.py   # Data splitting script
│   ├── dataset.py      # PyTorch Dataset & Dataloaders
│   ├── model.py        # CNN Architecture
│   └── train.py        # Training Loop
├── .gitignore          # Files to exclude from Git
├── README.md           # Project documentation
└── requirements.txt    # List of dependencies
```
## ⚙️ Installation
Clone the repository (or download the files).

Create a Virtual Environment:
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```
Install Dependencies: Use the provided requirements.txt to install the exact versions used in development. Note: We include the index URL to ensure the CUDA-enabled version of PyTorch is installed.
```bash
pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
## 🚀 How to Run

### 1. Data Preparation
Download the [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and place the `PetImages` folder inside `data/`.

Then, run the setup script to split data into Train/Val folders:
```bash
cd src
python data_setup.py
```
### 2. Training the Model
Train the CNN on your GPU. The best model will be saved to `models/cat_dog_model.pth`.
```bash
python train.py
```

### 3. Launch the Web App
Start the Streamlit interface to test the model:
```bash
streamlit run app.py
```
