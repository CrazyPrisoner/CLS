import os
import cv2
import numpy
import pandas
import random
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import img_as_float

train_data = './data/train_image/'
test_data = './data/test_image/'
data_csv = './data/data.csv'
dataframe_dm = pandas.read_csv(data_csv)
dataframe_dm.set_index('patientId', inplace=True)

def one_hot_label(target):
    if target == "Normal":
        ohl = numpy.array([1, 0, 0])
    elif target == "Lung Opacity":
        ohl = numpy.array([0, 1, 0])
    elif target == "No Lung Opacity / Not Normal":
        ohl = numpy.array([0, 0, 1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = numpy.float32(img)
        img = img.reshape(-1)       
        #print(img)
        #print(img.shape)
        train_images.append([numpy.array(img, dtype=numpy.float32), one_hot_label(dataframe_dm.class_tr[i.split(".")[0]])])
    random.shuffle(train_images)
    #print(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = numpy.float32(img)
        img = img.reshape(-1)       
        #print(img)
        #print(img.shape)
        test_images.append([numpy.array(img, dtype=numpy.float32), one_hot_label(dataframe_dm.class_tr[i.split(".")[0]])])
    random.shuffle(test_images)
    #print(test_images)
    return test_images

#train = train_data_with_label()
#print(train)

#test = test_data_with_label()
#print(test)