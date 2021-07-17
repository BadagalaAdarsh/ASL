import cv2
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('final_model.h5')

final_ans = {0:'A',1:'B',2:'C',3:'D',
            4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
            10:'K',11:'L',12:'M',13:'N',14:'O',
            15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
            21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'del',
            27:'nothing',28:'space' }

# img_path = '/home/adarsh/Jupyter_Notebook/asl_alphabet_test/R_test.jpg'

cap = cv2.VideoCapture(0)

def preprocess(img):
    image = cv2.resize(img,(64,64))
    test_image = (image[...,::-1])#.astype(np.float32)) 
    return test_image

while True:

    _, img = cap.read()
    im = img
    #im = cv2.flip(im,1)
    #im = cv2.flip(im,1)

    img = img[50:250, 80:280] # region of interest
    img = preprocess(img) 

    test_image = np.expand_dims(img, axis=0)
    test_image = test_image.reshape(1,64, 64,3)  
    #print(test_image)

    im=cv2.rectangle(im,(280,50),(80,250),(0,255,0),2)

    try:
        #result = model.predict(test_image, batch_size=1)
        pred = model.predict(test_image, batch_size = 1)
        result = np.argmax(pred, axis = 1)
#pred = np.argmax(pred, axis=1)
        result = result[0]
        #print(result)

        cv2.putText(im,final_ans[result],(250,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    except:
        pass

    cv2.imshow('main cam', im)
    #cv2.imshow('side cam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# image = image.load_img('M_test.jpg', target_size = (200,200))
# image = np.asarray(image)
# #image /= 255

# test_image = np.expand_dims(image, axis=0)


# test_image = test_image.reshape(1,200, 200,3)    
# print(test_image)

# result = model.predict(test_image, batch_size=1)
# result = result[0]

# result, = np.where(np.isclose(result, 1))
# result = result[0]

# final_ans = {0:'A',1:'B',2:'C',3:'D',
#             4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
#             10:'K',11:'L',12:'M',13:'N',14:'O',
#             15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
#             21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'del',
#             27:'nothing',28:'space' }


# print(final_ans[result])


