# TensorRT Cookbook in Chinese

## 06-PluginAndParser —— 结合使用 Parser 与 Plugin 的样例
+ 需要用到 onnx-graphsurgeon，可参考 08-Tool/onnxGraphSurgeon

### pyTorch-FailConvertNonZero
+ 在 .pt 转 .onnx 转 .plan 的过程中，转换 NonZero 节点失败的例子（TensorRT 不原生支持该算子）
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./pyTorch-FailConvertNonZero
pip install -r requirements.txt
python main.py
```
+ 参考输出结果，见 ./pyTorch-FailConvertNonZero/result.txt

### pyTorch-LayerNorm
+ 在 pyTorch 转 onnx 转 TensorRT 的过程中,替换一个 LayerNorm 以提高效率
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./pyTorch-LayerNorm
pip install -r requirements.txt
python main.py
```
+ 参考输出结果，见 ./pyTorch-LayerNorm/result.txt

### TensorFlow-AddScalar
+ 在 .pb 转 .onnx 转 .plan 的过程中，将 Add 算子替换为 Plugin 的例子（05-Plugin 中的 AddScalarPlugin）
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TensorFlow-addScalar
pip install -r requirements.txt
python TensorFlowToTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-addScalar/result.txt

