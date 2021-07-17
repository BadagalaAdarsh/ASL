import cv2
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
from keras import utils, callbacks
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
#from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



cap = cv2.VideoCapture(0)
model = load_model('final_model.h5')

test = ImageDataGenerator(rescale=1./255).flow_from_directory("/home/adarsh/Jupyter_Notebook/ASL/test_ind", 
                                                            target_size=(64, 64), class_mode=None)
pred = model.predict(test)
pred = np.argmax(pred, axis=1)
print(pred)



