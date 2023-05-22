#!/usr/bin/env python
# coding: utf-8

# In[4]:


import mediapipe as mp
import pandas as pd
import cv2
import tensorflow as tf


# In[2]:


def preprocess_image(path):
    #used only during creating the landmark dataset for svm
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


# In[3]:


def get_landmarks_list(image,results):
    for hand_landmarks in (results.multi_hand_landmarks):
        w,h = image.shape[:2]
        landmarks_x = []
        landmarks_y = []
        landmarks = []
        for _, landmark in enumerate(hand_landmarks.landmark):
#             landmarks_x.append(landmark.x)
#             landmarks_y.append(landmark.y)
            landmarks.append([landmark.x,landmark.y])
    return landmarks


# In[ ]:


def preprocess_landmark(landmarks_list,normalize=True):
    from nltk import flatten

    # Return relative distance with respect to first point
    base_x,base_y = landmarks_list[0][0],landmarks_list[0][1]
    relative_landmarks = []
    temp_landmark_x = 0
    temp_landmark_y = 0
    for landmark in landmarks_list:
        temp_landmark_x = landmark[0] - base_x
        temp_landmark_y = landmark[1] - base_y
        relative_landmarks.append([temp_landmark_x,temp_landmark_y])
    
    # Convert to one dimensional list
    relative_landmarks = flatten(relative_landmarks)
        
    if normalize:
        # Normalize || absolute value is considered for max value in the list
        max_value = max(list(map(abs, relative_landmarks)))
        relative_landmarks = list(map(lambda x: x/max_value,relative_landmarks))
    
    return relative_landmarks


# In[5]:


def get_bbox(cx_data,cy_data,delta=50):
    xmax=max(cx_data)+delta
    xmin=min(cx_data)-delta
    ymax=max(cy_data)+delta
    ymin=min(cy_data)-delta
    bbox = xmin,ymin,xmax,ymax
    return bbox


# In[6]:


def return_alphabet(number):
    return dict_alphabet[number]


# In[ ]:




