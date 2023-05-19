import numpy as np
import pandas as pd
import cv2
import seaborn as sb
import matplotlib.pyplot as mpp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps
import ssl, os, time

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

if(not os.environ.get("PYTHONHTTPSVERIFY", "") and 
   getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10, test_size = 2500, train_size = 7500)

X_train_scale = X_train/255
X_test_scale = X_test/225

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)

y_prediction = clf.predict(X_test_scale)

accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]: bottom_right[0]]

        imgpil = Image.fromarray(roi)
        imgbw = imgpil.convert("L")
        imgbwresize = imgbw.resize((28,28), Image.ANTIALIAS)
        imgbwresizeinvert = PIL.ImageOps.invert(imgbwresize)

        pixelfactor = 20
        minpixel = np.percentile(imgbwresizeinvert, pixelfactor)

        imgbwresizeinvert_scale = np.clip(imgbwresizeinvert - minpixel, 0, 255)
        
        maxpixel = np.max(imgbwresizeinvert)

        imgbwresizeinvert_scale = np.asarray(imgbwresizeinvert_scale)/maxpixel

        test_sample = np.array(imgbwresizeinvert_scale).reshape(1,784)
        test_prediction = clf.predict(test_sample)

        print(test_prediction)

        cv2.imshow("frame", gray)

        if cv2. waitKey(1) & 0xFF == ord("q"):
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()