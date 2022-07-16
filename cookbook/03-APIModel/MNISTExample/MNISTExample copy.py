#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
import calibrator

dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append(dataPath)
import loadMnistData

np.random.seed(97)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(97)
nTrainbatchSize = 128
paraFile = './paraTF.npz'
trtFile = "./model.plan"
inputImage = dataPath + '8.png'


# for FP16 mode
isFP16Mode = False
# for INT8 model
isINT8Mode = False
calibrationDataPath = dataPath + "test/"
calibrationCount = 1
cacheFile = "./int8.cache"

os.system("rm -rf ./paraTF.npz ./model.plan")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

b1 = tf.compat.v1.get_variable('b1', shape=[32], initializer=tf.constant_initializer(value=0.1))
h1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
h2 = h1 + b1
h3 = tf.nn.relu(h2)
h4 = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.compat.v1.get_variable('w2', shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b2 = tf.compat.v1.get_variable('b2', shape=[64], initializer=tf.constant_initializer(value=0.1))
h5 = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding='SAME')
h6 = h5 + b2
h7 = tf.nn.relu(h6)
h8 = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w3 = tf.compat.v1.get_variable('w3', shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
b3 = tf.compat.v1.get_variable('b3', shape=[1024], initializer=tf.constant_initializer(value=0.1))
h9 = tf.reshape(h8, [-1, 7 * 7 * 64])
h10 = tf.matmul(h9, w3)
h11 = h10 + b3
h12 = tf.nn.relu(h11)

tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=tfConfig)
sess.run(tf.compat.v1.global_variables_initializer())

mnist = loadMnistData.MnistData(dataPath, isOneHot=True)
for i in range(1000):
    xSample, ySample = mnist.getBatch(nTrainbatchSize, True)
    trainStep.run(session=sess, feed_dict={x: xSample, y_: ySample})
    if i % 100 == 0:
        train_acc = acc.eval(session=sess, feed_dict={x: xSample, y_: ySample})
        print("%s, step %d, acc = %f" % (dt.now(), i, train_acc))

xSample, ySample = mnist.getBatch(1000, False)
print("%s, test acc = %f" % (dt.now(), acc.eval(session=sess, feed_dict={x: xSample, y_: ySample})))

tfPara = {}  # 保存权重
#print("Weight:")
for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
    name, value = i.name, sess.run(i)
    #print(name,value.shape)
    tfPara[name] = value
np.savez("paraTF.npz", **tfPara)


# TensorRT 中重建并创建engine
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 3 << 30
    if isFP16Mode:
        config.flags = 1 << int(trt.BuilderFlag.FP16)
    else:
        config.flags = 1 << int(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, 28, 28), cacheFile)