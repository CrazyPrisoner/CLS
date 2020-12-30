import os
import cv2
import glob
import numpy
import ntpath
import pandas
import random
import datetime
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import img_as_float

from grpc.beta import implementations
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'inception_inference service host:port')
FLAGS = tf.app.flags.FLAGS

data = '/home/deka/Desktop/ML/CNN/data/test_flow/*'

def train_data_with_label():
    images = []
    list_of_files = glob.glob(data)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    path = os.path.join(latest_file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = numpy.float32(img)
    img = img.reshape(-1)
    images.append([numpy.array(img, dtype=numpy.float32)])
    print(images)
    return images, path

def get_metadata(path):
    metadata = {}
    file = ntpath.basename(path)
    file = os.path.join(file)
    name_and_time = file.split('.')[0]
    name = name_and_time.split('_')[0]
    time1 = name_and_time.split('_')[1]
    time = int(time1[:-3])
    date_time = datetime.datetime.fromtimestamp(time)
    print(name)  # name
    print(date_time) # date and time
    metadata['name'] = name
    metadata['dateandtime'] = date_time
    return metadata

def main(_):
    # Wrap bitstring in JSON
    training_images, path = train_data_with_label()
    tr_img_data = numpy.array([training_images[0]])
    metadata = get_metadata(path)
    print(metadata)
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.signature_name = 'predict'
    request.model_spec.name = 'classify'
    request.inputs['input'].dtype = types_pb2.DT_INT32
    #request.inputs['inputs'].float_val.append(feed_value2)
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(tr_img_data.astype(dtype=numpy.float32)))
    request.inputs['prob'].CopyFrom(tf.contrib.util.make_tensor_proto(0.8))
    request.output_filter.append('output')
    # Send request
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    prediction = stub.Predict(request, 5.0)  # 5 secs timeout
    floats = prediction.outputs['output'].float_val
    pred_arr = numpy.array(floats)
    #pred_arr = pred_arr.reshape(-1, 3)
    #pred_df = pandas.DataFrame(columns = ['normal', 'pb1', 'pb2'], data=pred_arr)
    print(pred_arr)
    #print(pred_df)


if __name__ == '__main__':
    tf.app.run()
