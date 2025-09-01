# Automatic Crater & Boulder Detection on Planetary Surfaces 🪐

![GitHub repo top image](https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Mars_crater.max-1000x1000.png)

This project provides an automated system to detect and classify craters and boulders on planetary surfaces from high-resolution **OHRC (Orbiter High-Resolution Camera)** images. It leverages deep learning techniques, specifically **Convolutional Neural Networks (CNNs)**, to perform accurate feature extraction and analysis.

## 🎯 Objective

The primary goal is to create a robust model that can automatically identify geological features like craters and boulders from satellite imagery. This automation is crucial for:
* **Improving surface exploration** and geological mapping.
* Assisting in **resource identification** on other planets.
* Supporting space missions by providing valuable data for **landing site selection** and safe terrain navigation.

---

## ✨ Key Features

* **Automated Detection:** Uses AI/ML to eliminate the need for manual, time-consuming analysis.
* **High Accuracy:** Employs modern object detection and classification techniques for reliable results.
* **Scalable:** Capable of processing large volumes of high-resolution satellite images efficiently.
* **Data-Driven:** Trained on labeled datasets of craters and boulders to ensure model robustness.

---

## 🛠️ Methodology

The detection process is built upon a standard deep learning pipeline for object detection:

1.  **Data Collection:** Utilizes high-resolution planetary images, such as those from an OHRC.
2.  **Image Preprocessing:** Images are normalized, resized, and cleaned to prepare them for the model.
3.  **Data Augmentation:** To improve model generalization and prevent overfitting, various augmentation techniques (e.g., rotation, flipping, brightness adjustments) are applied to the training dataset.
4.  **Model Training:** A Convolutional Neural Network (CNN) is trained on the preprocessed and augmented dataset. The model learns to distinguish the unique features of craters and boulders.
5.  **Inference & Detection:** The trained model is used to predict bounding boxes and classify craters and boulders in new, unseen images.



---

## 🚀 Tech Stack

* **Programming Language:** Python
* **Core Libraries:**
    * [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/) - For building and training the deep learning model.
    * [OpenCV](https://opencv.org/) - For image preprocessing and data augmentation.
    * [NumPy](https://numpy.org/) - For efficient numerical operations.
    * [Matplotlib](https://matplotlib.org/) - For visualizing results.

---

## ⚙️ Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/Crater-and-bouder-detection.git](https://github.com/your-username/Crater-and-bouder-detection.git)
    cd Crater-and-bouder-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage

To run detection on a new image, use the `detect.py` script.

```sh
python detect.py --image_path /path/to/your/image.tif --model_path /path/to/trained_model.h5
