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

import ctypes
import numpy as np
from cuda import cudart
import tensorrt as trt

np.random.seed(97)
nIn, cIn, hIn, wIn = 4, 3, 128, 128
data = np.random.rand(nIn * cIn * hIn * wIn).astype(np.float32).reshape(nIn, cIn, hIn, wIn)

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile0 = builder.create_optimization_profile()
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30

inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, [-1, cIn, hIn, wIn])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
network.mark_output(layer.get_output(0))

profile0.set_shape(inputT0.name, (1, cIn, hIn, wIn), (nIn, cIn, hIn, wIn), (nIn * 2, cIn, hIn, wIn))
profile1.set_shape(inputT0.name, (1, cIn, hIn, wIn), (nIn, cIn, hIn, wIn), (nIn * 2, cIn, hIn, wIn))
config.add_optimization_profile(profile0)
config.add_optimization_profile(profile1)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
_, stream0 = cudart.cudaStreamCreate()
_, stream1 = cudart.cudaStreamCreate()
context0 = engine.create_execution_context()
context1 = engine.create_execution_context()
context0.set_optimization_profile_async(0, stream0)
context1.set_optimization_profile_async(1, stream1)
context0.set_binding_shape(0, [nIn, cIn, hIn, wIn])
context1.set_binding_shape(2, [nIn, cIn, hIn, wIn])
print("Context0 binding all? %s" % (["No", "Yes"][int(context0.all_binding_shapes_specified)]))
print("Context1 binding all? %s" % (["No", "Yes"][int(context1.all_binding_shapes_specified)]))
for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context0.get_binding_shape(i), context1.get_binding_shape(i))

inputH0 = np.ascontiguousarray(data.reshape(-1))
inputH1 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context0.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
outputH1 = np.empty(context1.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))

_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream0)
_, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream1)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream0)
_, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream1)

for _ in range(5):
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)
    context0.execute_async_v2([int(inputD0), int(outputD0), int(0), int(0)], stream0)
    context1.execute_async_v2([int(0), int(0), int(inputD1), int(outputD1)], stream1)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)

cudart.cudaStreamSynchronize(stream0)
cudart.cudaStreamSynchronize(stream1)

print("check result:", np.all(outputH0 == outputH1))

cudart.cudaStreamDestroy(stream0)
cudart.cudaStreamDestroy(stream1)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
cudart.cudaFree(inputD1)
cudart.cudaFree(outputD1)
