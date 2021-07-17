import cv2
import numpy as np
from tensorflow.keras.models import load_model


def getletter(result):

    class_labels = {0:'A',1:'B',2:'B',3:'D',
                4:'E',5:'F',6:'G',7:'H',8:'I',9:'j',
                10:'K',11:'L',12:'M',13:'N',14:'O',
                15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
                21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'noting'}

    try:
        res = int(result)
        return class_labels[res]
    except:
        pass



# img_path = '/home/adarsh/Jupyter_Notebook/asl_alphabet_test/R_test.jpg'

cap = cv2.VideoCapture(0)
model = load_model('final_model.h5')

# def preprocess(img):
#     image = cv2.resize(img,(28,28))
#     test_image = (image[...,::-1])#.astype(np.float32)) 
#     return test_image

# while True:

#     _, img = cap.read()
#     im = img
#     #im = cv2.flip(im,1)

#     img = img[50:250, 380:580] # region of interest
#     img = preprocess(img) 

#     test_image = np.expand_dims(img, axis=0)
#     test_image = test_image.reshape(1,28, 28,1)  
#     #print(test_image)

#     im=cv2.rectangle(im,(580,50),(380,250),(0,255,0),2)

while True:

    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)

    roi = frame[100:400, 320:620]
    #cv2.imshow('roi', roi)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #roi = cv2.imread('./train/0/1.jpg')
    roi = cv2.resize(roi, (64,64), interpolation = cv2.INTER_AREA)

    #cv2.imshow('scaled image', roi)

    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0),5)

    roi = roi.reshape(1, 64, 64, 3)
    print(str(model.predict_classes(roi, 1, verbose = 0)[0]))
    result = str(model.predict_classes(roi, 1, verbose = 0)[0])
    cv2.putText(copy, getletter(result), (300,100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    #cv2.imshow('frame',frame)
    cv2.imshow('frame', copy)
    if cv2.waitKey(1) == 13:
        break

