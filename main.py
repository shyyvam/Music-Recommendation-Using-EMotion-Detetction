#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:58:56 2021

@author: shivam
"""

git clone https://github.com/misbah4064/emotion_recognition.git   #Importing pre trained model already worked on FER2013
%cd emotion_recognition           

from google.colab import drive #Importing Drive for songs and images
drive.mount("/content/drive")

import numpy as np
from google.colab.patches import cv2_imshow
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
!pip install pysoundfile
!pip install bitstring
import IPython
from IPython.display import Audio, display
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = "display"

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


def emotion_recog(frame):
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # frame = cv2.imread("image1.jpg")
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)  #BGRformat #thickness
        #eyes are going to be in the region of image of gray
        roi_gray = gray[y: y+h, x: x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+100, y+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    #if else to suggest music
        mood = maxindex
        print (mood)
        print (emotion_dict[mood])
        
        # {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        if (mood==0): #Angry
           print("Brothers Anthem Song")
           display(Audio("/content/drive/MyDrive/Songs/anger/Brothers Anthem Full Video - Akshay Kumar,Sidharth MalhotraVishal DadlaniAjay-Atul.mp3", autoplay=True))
           print("\n Chak Lein De")
           display(Audio("/content/drive/MyDrive/Songs/anger/Full Video Chak Lein DeChandni Chowk To ChinaAkshay Kumar, Deepika PadukoneKailash Kher.mp3", autoplay=False))
           print("\nGet ready to fight")
           display(Audio("/content/drive/MyDrive/Songs/anger/Get Ready To Fight Full Video Song BAAGHI Tiger Shroff, Grandmaster Shifuji Benny Dayal.mp3", autoplay=False))
           print("\nSadda Haq")
           display(Audio("/content/drive/MyDrive/Songs/anger/Sadda Haq Full Video Song RockstarRanbir Kapoor.mp3", autoplay=False))
           break
        elif (mood==1):  #Disgusted
            (IPython.display.Audio('Im_Superman.wav'))
            break
        elif (mood==2): #Fearful
           print("\nDark Piano")
           display(Audio("/content/drive/MyDrive/Songs/fearful/Dark Piano - Sociopath.mp3", autoplay=True))
           print("\nfear")
           display(Audio("/content/drive/MyDrive/Songs/fearful/Fear.mp3", autoplay=False))
           print("\nMelancholia creep song")
           display(Audio("/content/drive/MyDrive/Songs/fearful/MELANCHOLIA Music Box Sad, creepy song.mp3", autoplay=False))
           break
       
        elif (mood==3): #Happy
        
           print("\nHar Funn Maula")
           display(Audio("/content/drive/MyDrive/Songs/happy/Har Funn Maula (Video Song) Koi Jaane Na Aamir Khan Elli A Vishal D Zara K Tanishk B Amita.mp3", autoplay=True))
           print("\nBeliever")
           display(Audio("/content/drive/MyDrive/Songs/happy/Iron man - Believer.mp3", autoplay=False))
           print("\nMATARGASHTI")
           display(Audio("/content/drive/MyDrive/Songs/happy/MATARGASHTI full VIDEO Song TAMASHA Songs 2015 Ranbir Kapoor, Deepika Padukone T-Series.mp3", autoplay=False))
           print("\nlove you")
           display(Audio("/content/drive/MyDrive/Songs/happy/love you.mp3", autoplay=False))
            
        elif (mood==4): #Neutral
           print("\nHate How much I love you")
           display(Audio("/content/drive/MyDrive/Songs/neutral/Conor Maynard - Hate How Much I Love You (Official Video).mp3", autoplay=True))
           print("\nNatural")
           display(Audio("/content/drive/MyDrive/Songs/neutral/Imagine Dragons - Natural (Lyrics).mp3", autoplay=False))
           print("\nEarth")
           display(Audio("/content/drive/MyDrive/Songs/neutral/Lil Dicky - Earth (Official Music Video).mp3", autoplay=False))
           print("\n7 years")
           display(Audio("/content/drive/MyDrive/Songs/neutral/Lukas Graham - 7 Years [Official Music Video].mp3", autoplay=False))
          
           break
        elif (mood==5): #Sad
           print("\n Baarish ki jaaye")
           display(Audio("/content/drive/MyDrive/Songs/sad/Baarish Ki Jaaye B Praak Ft Nawazuddin Siddiqui & Sunanda Sharma Jaani Arvindr Khaira DM.mp3", autoplay=True))
           print("\nFILHAAL")
           display(Audio("/content/drive/MyDrive/Songs/sad/FILHALL Akshay Kumar Ft Nupur Sanon BPraak Jaani Arvindr Khaira Ammy Virk Official V.mp3", autoplay=False))
           print("Teri Mitti")
           display(Audio("/content/drive/MyDrive/Songs/sad/Teri Mitti - LyricalKesariAkshay Kumar & Parineeti ChopraArkoB Praak Manoj Muntashir.mp3", autoplay=False))
           break
      
        elif (mood==6): #Surprised
           print("\nNatural")
           display(Audio("/content/drive/MyDrive/Songs/surprise/Imagine Dragons - Natural (Lyrics) (1).mp3", autoplay=True))
           print("\nShell Shocked")
           display(Audio("/content/drive/MyDrive/Songs/surprise/Juicy J, Wiz Khalifa, Ty Dolla $ign - Shell Shocked feat Kill The Noise & Madsonik (Official Vid.mp3", autoplay=False))
           print("\nMC sher Gully boy")
           display(Audio("/content/drive/MyDrive/Songs/surprise/MC SHER super hit rapping songin gully boy film (1).mp3", autoplay=False))
           print("\nSugar")
           display(Audio("/content/drive/MyDrive/Songs/surprise/Maroon 5 - Sugar (Official Music Video).mp3", autoplay=False))
           break

    # cv2_imshow(frame)
    return frame