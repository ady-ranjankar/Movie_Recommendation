
# coding: utf-8

# In[39]:

import os
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import glob
import imageio
import dlib
import math
import pickle
from imutils import face_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


# In[40]:


class Mood:
    def detect():
        data1 = pd.read_csv('MoviesGenreFin.csv', names=['Movie','Genre'], header=None)
        data1=np.asarray(data1)
        model = pickle.load(open('C:/Users/adity/Downloads/finalized_model.sav', 'rb'))
    # initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
        p = "C:/Users/adity/Downloads/shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)


        t=[]
        for filepath in glob.iglob(r'C:/Users/adity/instance/htmlfi/*.jpg'):
            file=filepath
        file=file.replace("\\","/")
        image = imageio.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        a=[]
        j=[]
        for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                a.append(shape)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # show the output image with the face detections + facial landmarks

        a=np.asarray(a)
        if(np.shape(a)==(1,68,2)):
            a=a.reshape(68,2)
            l=0
            m=0
                #print(a)
            for i in range(0,68):
                for k in range(0,68):
                    l=a[i][0]-a[k][0]
                    l=l*l
                    m=a[i][1]-a[k][1]
                    m=m*m
                    j.append(math.sqrt(m+l))


            j=np.asarray(j)
            t.append(j)
        else:
            m="Irrelevant"
            b=['Input Correct Image']
            return m,b
        t=np.asarray(t)
        t.shape
        test=t.reshape(1,-1)
        mood=["Anger","Fear","Sadness","Surprised","Disgust","Joy"]
        m=int(model.predict(test))
        m=mood[m]

        if(m=="Anger"):
            k=['Family Film','Comedy']
        elif(m=="Fear"):
            k=['Adventure','Romance Film']
        elif(m=="Sadness"):
            k=['Drama','Comedy','Animation']
        elif(m=='Surprised'):
            k=['Thriller','Crime Fiction']
        elif(m=='Disgust'):
            k=['Family Film','Adventure']
        else:
            k=['Thriller','Comedy']


        b=[]
        q=0
        
        data1, X_test = train_test_split(data1, test_size=0)
        for i in data1:
            if(i[1] in k):   
                b.append(i[0])
                q=q+1
            if(q==10):
                break
        src=file
        dst='C:/Users/adity/instance/Stored Images'
        shutil.move(src, dst)
        return m,b

