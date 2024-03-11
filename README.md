# Automatic Number Plate Recognition System
This project aims to develop an Automatic Number Plate Recognition (ANPR) system using computer vision techniques. The system is capable of accurately detecting license plates from images, segmenting characters from the detected plates, and recognizing the characters using Optical Character Recognition (OCR) methods.

# Expected Outcome
The expected outcome is to develop an ANPR system that can:

* Detect license plates from input images.
* Segment individual characters from the detected license plates.
* Recognize the characters using OCR techniques.

# Dataset
The following datasets are used for training and testing the ANPR system:

* Indian License Plates
* Indian Vehicle License Plate Dataset
These datasets consist of images containing vehicles with visible license plates, including various scenarios such as different lighting conditions, angles, and vehicle types to ensure the robustness of the system.

Additionally, a separate dataset containing 36 subfolders, each containing training images of characters from 0 to 9 and A to Z, is used to train the CNN OCR model.

# Approach
The ANPR system is developed using the following approach:

* License Plate Detection
A function detect_plate is implemented to detect license plates in images using Haar cascades. Another function display is used to display images with detected plates.

* Image Processing and Display
The input image is loaded, and the license plate detection function is applied to detect plates in the image. The input image with the detected license plate is then displayed.

* Character Segmentation
Functions find_contours and segment_characters are defined to find contours and segment individual characters from the detected license plate image, respectively.

### Optical Character Recognition (OCR)
Two OCR methods are employed:

 ##### EasyOCR:
   A library for text recognition that directly extracts text from segmented characters.
 ##### Convolutional Neural Network (CNN):
   A CNN model is trained on a dataset of character images for character recognition. The CNN involves dataset preparation, model training with data augmentation, and character prediction using softmax outputs.
The choice between EasyOCR and the CNN model depends on the trade-off between simplicity and accuracy, considering project requirements and resource availability.

# Future Plans
Expand the project to ANPR from video data, enabling real-time surveillance and detection of license plates of moving vehicles.
Implement algorithms for object tracking and frame-by-frame analysis to extract license plate information continuously.
Integrate with existing traffic management systems for automatic detection of traffic violations and real-time alerts to authorities.
Incorporate machine learning models trained on diverse datasets to improve accuracy and robustness in challenging conditions.
Market the system as a comprehensive surveillance solution for traffic management, revolutionizing law enforcement and enhancing public safety on roads.
Getting Started
To get started with this project, follow these steps:
