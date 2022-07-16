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
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
sys.path.append( dataPath )
import loadMnistData

np.random.seed(97)
tf.compat.v1.set_random_seed(97)
nTrainbatchSize = 128
pbFile = "./model-NHWC(C=2).pb"
onnxFile = "./model-NHWC(C=2).onnx"
trtFile = "./model-NHWC(C=2).plan"
inputImage = dataPath + '8.png'

os.system("rm -rf ./model.pb ./model.onnx ./model.plan")
np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
cudart.cudaDeviceSynchronize()

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
x   = tf.compat.v1.placeholder(tf.float32, [None,28,28,2], name='x')
y_  = tf.compat.v1.placeholder(tf.float32, [None,10], name='y_')

w1  = tf.compat.v1.get_variable('w1', shape = [5,5,1,32], initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))
b1  = tf.compat.v1.get_variable('b1', shape = [32], initializer = tf.constant_initializer(value=0.1))
h1  = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
h2  = h1 + b1
h3  = tf.nn.relu(h2)
h4  = tf.nn.max_pool2d(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2  = tf.compat.v1.get_variable('w2', shape = [5,5,32,64], initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))
b2  = tf.compat.v1.get_variable('b2', shape = [64], initializer = tf.constant_initializer(value=0.1))
h5  = tf.nn.conv2d(h4, w2, strides=[1, 1, 1, 1], padding='SAME')
h6  = h5 + b2
h7  = tf.nn.relu(h6)
h8  = tf.nn.max_pool2d(h7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w3  = tf.compat.v1.get_variable('w3', shape = [7*7*64,1024], initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))
b3  = tf.compat.v1.get_variable('b3', shape = [1024], initializer = tf.constant_initializer(value=0.1))
h9  = tf.reshape(h8, [-1, 7*7*64])
h10 = tf.matmul(h9, w3)
h11 = h10 + b3
h12 = tf.nn.relu(h11)

w4  = tf.compat.v1.get_variable('w4', shape = [1024,10], initializer = tf.truncated_normal_initializer(mean=0,stddev=0.1))
b4  = tf.compat.v1.get_variable('b4', shape = [10], initializer = tf.constant_initializer(value=0.1))
h13 = tf.matmul(h12, w4)
h14 = h13 + b4
y   = tf.nn.softmax(h14, name='y')
z   = tf.argmax(y,1,name='z')

crossEntropy    = -tf.reduce_sum(y_*tf.math.log(y))
trainStep       = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(crossEntropy)

output          = tf.argmax(y,1)
resultCheck     = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
acc             = tf.reduce_mean(tf.cast(resultCheck, tf.float32), name='acc')

tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=tfConfig)
sess.run(tf.compat.v1.global_variables_initializer())

mnist = loadMnistData.MnistData(dataPath, isOneHot=True)
for i in range(1000):
    xSample, ySample = mnist.getBatch(nTrainbatchSize,True)
    xSample = np.tile(xSample,[1,1,1,2])
    trainStep.run(session = sess, feed_dict={x: xSample, y_: ySample})
    if i%100 == 0:
        train_acc = acc.eval(session = sess, feed_dict={x:xSample, y_: ySample})
        print("%s, step %d, acc = %f"%(dt.now(), i, train_acc))

xSample, ySample = mnist.getBatch(100,False)
xSample = np.tile(xSample,[1,1,1,2])
print( "%s, test acc = %f"%(dt.now(), acc.eval(session = sess, feed_dict={x:xSample, y_: ySample})) )

constantGraph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,['z'])
with tf.gfile.FastGFile(pbFile, mode='wb') as f:
    f.write(constantGraph.SerializeToString())
sess.close()
print("Succeeded building model in TensorFlow!")

# 将 .pb 文件转换为 .onnx 文件 ----------------------------------------------------
os.system("python -m tf2onnx.convert --input \"%s\" --output \"%s\" --inputs 'x:0' --outputs 'z:0'"%(pbFile,onnxFile))
print("Succeeded converting model into onnx!")

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder                     = trt.Builder(logger)
    network                     = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile                     = builder.create_optimization_profile()
    config                      = builder.create_builder_config()
    config.flags                = 1<<int(trt.BuilderFlag.FP16)          # now use fp16 mode, comment this line to use fp32 mode
    config.max_workspace_size   = 3<<30
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding .onnx file!")
        exit()
    print("Succeeded finding .onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse( model.read() ):
            print ("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            exit()
        print ("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    inputTensor.shape = [-1,28,28,2]
    profile.set_shape(inputTensor.name, (1,28,28,2),(4,28,28,2),(16,28,28,2))
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network,config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write( engineString )
    engine = trt.Runtime(logger).deserialize_cuda_engine( engineString )

context = engine.create_execution_context()
context.set_binding_shape(0,[1,28,28,2])
_, stream   = cudart.cudaStreamCreate()
print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0));
print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1));

data        = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE).astype(np.float32)
data        = np.tile(data,[1,2,1,1]).transpose(0,2,3,1)
inputH0     = np.ascontiguousarray(data.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
_,inputD0   = cudart.cudaMallocAsync(inputH0.nbytes,stream)
_,outputD0  = cudart.cudaMallocAsync(outputH0.nbytes,stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
