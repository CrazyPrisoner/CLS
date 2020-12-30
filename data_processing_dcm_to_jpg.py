import os
import cv2
import numpy
import pandas
import pydicom
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

det_class_path = './data/stage_2_detailed_class_info.csv'
bbox_path = './data/stage_2_train_labels.csv'
dicom_dir = './data/stage_2_train_images/'
det_class_df = pandas.read_csv(det_class_path)
bbox_df = pandas.read_csv(bbox_path)
comb_bbox_df = pandas.merge(bbox_df, det_class_df, how='inner', on='patientId')
comb_bbox_df = pandas.concat([bbox_df, 
                        det_class_df.drop('patientId',1)], 1)

comb_bbox_df = comb_bbox_df.drop_duplicates(subset="patientId", keep="last")
comb_bbox_df.to_csv('./data/data.csv', index=False)
print("DATA LEN:",len(comb_bbox_df))
id_list_dcm = []

for row in range(len(comb_bbox_df)):
    id_list_dcm.append(comb_bbox_df.patientId[row])

#print(id_list_train)
image_path = "./data/train_image"
dcm_path = "./data/dicom_files"
PNG = False
print(len(id_list_dcm))
for n, image in enumerate(id_list_dcm):
    image = image + ".dcm"
    dc_file = pydicom.dcmread(os.path.join(dcm_path, image))
    rows = []
    pixel_array_np = dc_file.pixel_array
    if PNG == False:
        image = image.replace('.dcm', '.jpg')
    else:
        image = image.replace('.dcm', '.jpg')
    print(image)
    cv2.imwrite(os.path.join(image_path, image), pixel_array_np)
    if n % 50 == 0:
        print("{} image converted".format(n))
