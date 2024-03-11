#!/usr/bin/env python
# coding: utf-8

# # Automatic Number Plate Recognition System

# ### Team Members : 
# 
# - 1) Akshay S.G - 21110189 - AI & DS A
# - 2) Samyuktha S - 21011101106 - AI & DS B
# - 3) Raveesh R - 21011101099 - AI & DS B

# ### Expected Outcome:
# 
# - The expected outcome is to develop an Automatic Number Plate Recognition (ANPR) system using computer vision techniques. This system will be capable of accurately detecting license plates from images, segmenting characters from the detected plates, and recognizing the characters using Optical Character Recognition (OCR) methods.

# ### Details of the dataset to be used:
# 
# ##### Indian License Plates : https://www.kaggle.com/datasets/thamizhsterio/indian-license-plates
# 
# ##### Indian vehicle license plate dataset : https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset
# 
# - The dataset used for training and testing the ANPR system consists of images containing vehicles with visible license plates. This dataset includes various scenarios such as different lighting conditions, angles, and vehicle types to ensure the robustness of the system.
# 
# - To train the CNN OCR model, a separate dataset containing 36 subfolders, each containing training images of characters from 0 to 9 and A to Z, is used.

# ### Work done so far:
# 
# ANPR systems are crucial for various applications like automated toll collection, parking management, and law enforcement. Here's a brief description of the approach and steps involved:
# 
# ##### Library Import:
# 
# - Import necessary libraries such as OpenCV, Matplotlib, NumPy, TensorFlow, and EasyOCR.
# 
# ##### License Plate Detection:
# 
# - Define a function (detect_plate) to detect license plates in images using Haar cascades.
# - Implement a function (display) to display images with detected plates.
# 
# ##### Image Processing and Display:
# 
# - Load an input image.
# - Apply the license plate detection function to detect plates in the image.
# - Display the input image with the detected license plate.
# 
# ##### Character Segmentation:
# 
# - Define functions for finding contours (find_contours) and segmenting characters (segment_characters).
# - Segment individual characters from the detected license plate image.
# 
# ##### Optical Character Recognition (OCR):
# 
# - Two OCR methods are employed. EasyOCR, a library for text recognition, offers simplicity and efficiency by directly extracting text from segmented characters. 
# - In contrast, a Convolutional Neural Network (CNN) is trained on a dataset of character images, offering flexibility and potentially higher accuracy. 
# - The CNN involves dataset preparation, model training with data augmentation, and character prediction using softmax outputs. 
# - While EasyOCR requires minimal setup and offers language support, the CNN demands more effort but provides customization options and potentially better performance. 
# - The choice depends on trade-offs between simplicity and accuracy, considering project requirements and resource availability.

# ### Future Plans : 
# 
# - In the future, expanding this project to ANPR from video data is feasible and offers significant potential. By leveraging computer vision techniques, we can process video streams in real-time, enabling the surveillance camera to detect license plates of moving vehicles. 
# - Implementing algorithms for object tracking and frame-by-frame analysis would allow us to extract license plate information continuously. Integration with existing traffic management systems would enable automatic detection of traffic violations, such as speeding or running red lights. By providing real-time alerts to authorities, our surveillance camera can enhance road safety and enforce traffic regulations effectively. 
# - Additionally, incorporating machine learning models trained on diverse datasets would improve the system's accuracy and robustness, ensuring reliable performance even in challenging conditions such as varying lighting and weather. Ultimately, marketing this as a comprehensive surveillance solution for traffic management could revolutionize law enforcement and enhance public safety on roads.

# ## Import Libraries

# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# ## License Plate Detection:

# In[4]:


plate_cascade = cv2.CascadeClassifier('/home/akshay/SNU_AI_DS/6th_sem/CV/Project/ANPR/Trial2/indian_license_plate.xml')


# In[5]:


def detect_plate(img, text=''): 
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7) 
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] 
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) 
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)
        
    return plate_img, plate 


# In[6]:


def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


# ## Image Processing and Display:

# In[7]:


img = cv2.imread('/home/akshay/SNU_AI_DS/6th_sem/CV/Project/ANPR/Trial2/car.jpg')
display(img, 'input image')


# In[8]:


output_img, plate = detect_plate(img)


# In[9]:


display(output_img, 'detected license plate in the input image')


# In[10]:


display(plate, 'extracted license plate from the image')


# ## Character Segmentation:

# In[11]:


def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


# In[12]:


def segment_characters(image) :

    
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


# In[13]:


char = segment_characters(plate)


# In[14]:


for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')


# ## Optical Character Recognition (OCR)

# ### Using EasyOCR

# In[15]:


import easyocr
reader = easyocr.Reader(['ch_sim','en'])


# In[16]:


results = reader.readtext(plate)

if results:
    text = results[0][1]
else:
    text = "No text found"

print(" \n License Plate :", text)


# ### Using CNN

# In[17]:


import tensorflow.keras.backend as K
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = '/home/akshay/SNU_AI_DS/6th_sem/CV/Project/ANPR/data/data/data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  
        target_size=(28,28),  
        batch_size=1,
        class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
        path+'/val', 
        target_size=(28,28),  
        class_mode='sparse')


# In[18]:


def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)


# In[19]:


K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])


# In[20]:


model.summary()


# In[21]:


class stop_training_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_custom_f1score') > 0.99):
            self.model.stop_training = True


# In[22]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')


# In[28]:


batch_size = 1
model.fit_generator(
      train_generator,
      steps_per_epoch = train_generator.samples // batch_size,
      validation_data = validation_generator, 
      epochs = 100, verbose=1)


# In[29]:


def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return e_x / e_x.sum(axis=0)
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): 
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) 
        y_ = np.argmax(model.predict(img)[0], axis=-1)  # or use thresholding if binary
        character = dic[y_] # Using np.argmax to get the index of the maximum value in y_
        output.append(character)
        
    plate_number = ''.join(output)
    
    return plate_number

print(show_results())


# In[30]:


plt.figure(figsize=(10,6))
for i,ch in enumerate(char):
    img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
    plt.subplot(3,4,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted: {show_results()[i]}')
    plt.axis('off')
plt.show()


# In[33]:


print(show_results())

