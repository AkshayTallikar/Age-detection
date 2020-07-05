import pickle

import matplotlib
import os
import ktrain
from ktrain import vision as vis
from keras.preprocessing.image import image
from keras.metrics import categorical_accuracy
import os.path
from imutils import paths
import numpy as np
import re
import csv
import cv2
import pandas as pd
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from helpers import resize_to_fit
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, \
    BatchNormalization
import sklearn
import matplotlib.pyplot as plt
from csv import writer

LETTER_IMAGES_FOLDER1 = "001"
LETTER_IMAGES_FOLDER2 = "002"
LETTER_IMAGES_FOLDER3 = "003"
LETTER_IMAGES_FOLDER4 = "004"
LETTER_IMAGES_FOLDER5 = "005"
LETTER_IMAGES_FOLDER6 = "006"
LETTER_IMAGES_FOLDER7 = "007"
LETTER_IMAGES_FOLDER8 = "008"
LETTER_IMAGES_FOLDER9 = "009"
LETTER_IMAGES_FOLDER10 = "010"
LETTER_IMAGES_FOLDER11 = "011"
LETTER_IMAGES_FOLDER12 = "012"
LETTER_IMAGES_FOLDER13 = "013"
LETTER_IMAGES_FOLDER14 = "014"
LETTER_IMAGES_FOLDER15 = "015"
LETTER_IMAGES_FOLDER16 = "016"
LETTER_IMAGES_FOLDER17 = "017"
LETTER_IMAGES_FOLDER18 = "018"
LETTER_IMAGES_FOLDER19 = "019"
LETTER_IMAGES_FOLDER20 = "020"
LETTER_IMAGES_FOLDER21 = "021"
LETTER_IMAGES_FOLDER22 = "022"
LETTER_IMAGES_FOLDER23 = "023"
LETTER_IMAGES_FOLDER24 = "024"
LETTER_IMAGES_FOLDER25 = "025"
LETTER_IMAGES_FOLDER26 = "026"
LETTER_IMAGES_FOLDER27 = "027"
LETTER_IMAGES_FOLDER28 = "028"
LETTER_IMAGES_FOLDER29 = "029"
LETTER_IMAGES_FOLDER30 = "030"
LETTER_IMAGES_FOLDER31 = "031"
LETTER_IMAGES_FOLDER32 = "032"
LETTER_IMAGES_FOLDER33 = "033"
LETTER_IMAGES_FOLDER34 = "034"
LETTER_IMAGES_FOLDER35 = "035"
LETTER_IMAGES_FOLDER36 = "036"
LETTER_IMAGES_FOLDER37 = "037"
LETTER_IMAGES_FOLDER38 = "038"
LETTER_IMAGES_FOLDER39 = "039"
LETTER_IMAGES_FOLDER40 = "040"
LETTER_IMAGES_FOLDER41 = "041"
LETTER_IMAGES_FOLDER42 = "042"
LETTER_IMAGES_FOLDER43 = "043"
LETTER_IMAGES_FOLDER44 = "044"
LETTER_IMAGES_FOLDER45 = "045"
LETTER_IMAGES_FOLDER46 = "046"
LETTER_IMAGES_FOLDER47 = "047"
LETTER_IMAGES_FOLDER48 = "048"
LETTER_IMAGES_FOLDER49 = "049"
LETTER_IMAGES_FOLDER50 = "050"
LETTER_IMAGES_FOLDER51 = "051"
LETTER_IMAGES_FOLDER52 = "052"
LETTER_IMAGES_FOLDER53 = "053"
LETTER_IMAGES_FOLDER54 = "054"
LETTER_IMAGES_FOLDER55 = "055"
LETTER_IMAGES_FOLDER56 = "056"
LETTER_IMAGES_FOLDER57 = "057"
LETTER_IMAGES_FOLDER58 = "058"
LETTER_IMAGES_FOLDER59 = "059"
LETTER_IMAGES_FOLDER60 = "060"
LETTER_IMAGES_FOLDER61 = "061"
LETTER_IMAGES_FOLDER62 = "062"
LETTER_IMAGES_FOLDER63 = "063"
LETTER_IMAGES_FOLDER64 = "064"
LETTER_IMAGES_FOLDER65 = "065"
LETTER_IMAGES_FOLDER66 = "066"
LETTER_IMAGES_FOLDER67 = "067"
LETTER_IMAGES_FOLDER68 = "068"
LETTER_IMAGES_FOLDER69 = "069"
LETTER_IMAGES_FOLDER70 = "070"
LETTER_IMAGES_FOLDER71 = "071"
LETTER_IMAGES_FOLDER72 = "072"
LETTER_IMAGES_FOLDER73 = "073"
LETTER_IMAGES_FOLDER74 = "074"
LETTER_IMAGES_FOLDER75 = "075"
LETTER_IMAGES_FOLDER76 = "076"
LETTER_IMAGES_FOLDER77 = "077"
LETTER_IMAGES_FOLDER78 = "078"
LETTER_IMAGES_FOLDER79 = "079"
LETTER_IMAGES_FOLDER80 = "080"
LETTER_IMAGES_FOLDER81 = "081"
LETTER_IMAGES_FOLDER82 = "082"
LETTER_IMAGES_FOLDER83 = "083"
LETTER_IMAGES_FOLDER84 = "084"
LETTER_IMAGES_FOLDER85 = "085"
LETTER_IMAGES_FOLDER86 = "086"
LETTER_IMAGES_FOLDER87 = "087"
LETTER_IMAGES_FOLDER88 = "088"
LETTER_IMAGES_FOLDER89 = "089"
LETTER_IMAGES_FOLDER90 = "090"
LETTER_IMAGES_FOLDER91 = "091"
LETTER_IMAGES_FOLDER92 = "092"
LETTER_IMAGES_FOLDER93 = "093"
LETTER_IMAGES_FOLDER94 = "094"
LETTER_IMAGES_FOLDER95 = "095"
LETTER_IMAGES_FOLDER96 = "096"
LETTER_IMAGES_FOLDER97 = "097"
LETTER_IMAGES_FOLDER98 = "098"
LETTER_IMAGES_FOLDER99 = "099"
LETTER_IMAGES_FOLDER100 = "100"

images = []


def get_all_pics_and_create_csv_file(folder):
    for image_file in paths.list_images(folder):
        image = cv2.imread(image_file)
        imgresize = resize_to_fit(image, 64, 64)
        images.append(imgresize / 255.0)


def just_need_a_function():
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER1)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER2)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER3)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER4)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER5)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER6)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER7)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER8)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER9)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER10)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER11)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER12)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER13)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER14)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER15)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER16)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER17)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER18)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER19)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER20)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER21)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER22)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER23)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER24)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER25)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER26)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER27)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER28)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER29)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER30)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER31)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER32)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER33)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER34)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER35)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER36)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER37)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER38)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER39)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER40)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER41)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER42)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER43)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER44)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER45)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER46)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER47)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER48)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER49)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER50)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER51)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER52)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER53)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER54)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER55)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER56)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER57)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER58)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER59)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER60)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER61)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER62)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER63)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER64)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER65)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER66)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER67)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER68)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER69)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER70)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER71)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER72)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER73)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER74)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER75)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER76)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER77)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER78)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER79)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER80)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER81)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER82)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER83)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER84)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER85)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER86)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER87)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER88)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER89)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER90)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER91)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER92)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER93)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER94)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER95)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER96)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER97)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER98)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER99)
    get_all_pics_and_create_csv_file(LETTER_IMAGES_FOLDER100)


just_need_a_function()

images = np.array(images)
data = pd.read_csv('age.csv')
labels = np.array(data)
print(images.shape)
print(labels.shape)
images, labels = shuffle(images, labels)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(images, labels, test_size=0.1)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.2))

model.add(Conv2D(32, (3, 3), activation='relu', ))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.3))

model.add(Conv2D(64, (3, 3), activation='relu', ))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.4))

model.add(Conv2D(128, (3, 3), activation='relu', ))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(100, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

pred = []

folderss = 'test images'


def show_pred(folder):
    for image_file in paths.list_images(folder):
        image = cv2.imread(image_file)
        imgresize = resize_to_fit(image, 64, 64)
        pred.append(imgresize / 255.0)


show_pred(folderss)
pred = np.array(pred)
pred.reshape(-1, 1)
y_pred = model.predict(pred)
print(y_pred)
top3 = np.argmax(y_pred)
print(top3)
