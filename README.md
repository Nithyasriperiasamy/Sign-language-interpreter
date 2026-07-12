 CNN-Based Sign Language Translator

A real-time sign language translator that recognizes **American Sign Language (ASL)** alphabet gestures using a custom Convolutional Neural Network (CNN) and converts them into text. The application also includes an optional text-to-speech feature that reads the translated sentence aloud.

The project combines **TensorFlow/Keras**, **MediaPipe Hands**, **OpenCV**, and **Streamlit** to provide an interactive and user-friendly interface for real-time gesture recognition.



 Why I Built This Project

I wanted to explore how deep learning and computer vision can be combined to solve a real-world accessibility problem. Rather than using a pre-trained image classification model, I trained a CNN from scratch to better understand the complete workflow—from data preprocessing and model training to real-time deployment.

The project also gave me hands-on experience integrating machine learning models into a web application and optimizing the prediction pipeline for live webcam input.

Features

* Real-time hand detection using MediaPipe Hands
* Custom CNN trained on the ASL Alphabet dataset
* Webcam-based gesture recognition
* Image upload for testing predictions
* Live prediction with confidence score
* Sentence builder with Add, Delete, Space, and Clear options
* Text-to-speech support for translated sentences
* Prediction history panel
* Download translated text as a `.txt` file
* Responsive Streamlit interface with dark-mode support



Tech Stack

Languages

* Python
  
Libraries & Frameworks

* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy
* Streamlit



## Model Architecture

The model is a custom Convolutional Neural Network built using the Keras Sequential API.

```
Input (64 × 64 × 3)

↓
Conv2D (32) + ReLU
↓
MaxPooling2D

↓
Conv2D (64) + ReLU
↓
MaxPooling2D

↓
Conv2D (128) + ReLU

↓
Flatten

↓
Dense (256) + ReLU
↓
Dropout (0.5)

↓
Dense (29)
↓
Softmax
```

The model classifies 29 classes corresponding to the ASL alphabet, along with **space**, **delete**, and **nothing**.


## Training Pipeline

The training process includes several techniques to improve model performance and reduce overfitting.

* Image augmentation using `ImageDataGenerator`

  * Rotation
  * Width and height shifts
  * Shear transformation
  * Zoom
  * Brightness variation
* Early stopping to prevent unnecessary training
* Model checkpointing to save the best-performing model
* Learning rate reduction when validation performance plateaus
* Automatic generation of accuracy, loss, and confusion matrix plots

After training, the following files are saved inside the `models/` directory:

cnn_model.keras
class_names.txt
accuracy_plot.png
loss_plot.png
confusion_matrix.png


Prediction Workflow

1. Capture an image from the webcam or upload an image.
2. MediaPipe detects the hand region.
3. The detected hand is cropped with padding.
4. The cropped image is resized to **64 × 64**.
5. Pixel values are normalized.
6. The CNN predicts the gesture.
7. The application displays the predicted letter along with its confidence score.



Dataset

The model is trained using the **ASL Alphabet Dataset** available on Kaggle.

Classes

 A–Z
space
del
nothing

 Download

pip install kaggle

kaggle datasets download -d grassknoted/asl-alphabet

unzip asl-alphabet.zip -d dataset/


Expected directory structure:

dataset/
└── asl_alphabet_train/
    ├── A/
    ├── B/
    ├── ...
    ├── Z/
    ├── space/
    ├── del/
    └── nothing/


Project Structure

Sign-Language-Translator/
│
├── dataset/
├── models/
├── utils/
│   ├── preprocessing.py
│   ├── hand_detector.py
│   └── tts.py
│
├── assets/
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


Installation

Clone the repository and install the required dependencies.

git clone https://github.com/<your-username>/Sign-Language-Translator.git

cd Sign-Language-Translator

python -m venv venv
 Windows
venv\Scripts\activate

pip install -r requirements.txt

 Training the Model

Once the dataset has been downloaded, start training using:


python train.py --data-dir dataset/asl_alphabet_train --epochs 30


The trained model and evaluation plots will be saved automatically in the `models/` folder.



 Running the Application

Launch the Streamlit application using:

streamlit run app.py

Then open the local URL displayed in your terminal (typically `http://localhost:8501`).

 Deployment

The application can be deployed easily using **Hugging Face Spaces**.

1. Create a new Streamlit Space.
2. Upload the project files.
3. Add the trained model (`cnn_model.keras`) and `class_names.txt`.
4. Hugging Face automatically installs the dependencies from `requirements.txt` and launches the application.


What I Learned

Working on this project helped me strengthen my understanding of:

 Building CNN models for image classification
 Image preprocessing and augmentation techniques
 Hand detection using MediaPipe
 Real-time inference with OpenCV
 Deploying machine learning applications using Streamlit
 Integrating deep learning models into end-to-end applications
Future Improvements

Some ideas for extending this project include:

  Support dynamic ASL words and sentences using LSTM or Transformer-based models
  Recognize two-hand gestures
  Convert the model to TensorFlow Lite for edge deployment
  Improve robustness under different lighting conditions
  Add multilingual translation and text-to-speech support

cknowledgements

 ASL Alphabet Dataset (Kaggle)
 TensorFlow
 OpenCV
 MediaPipe
 Streamlit


