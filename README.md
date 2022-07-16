# 英伟达TensorRT加速AI推理Hackathon 2022 —— Transformer模型优化赛

## 大赛报名入口
点击[报名入口](https://tianchi.aliyun.com/competition/entrance/531953/information)，注册阿里云账号，报名参赛。

## 开发环境的设置命令
关于如何安装nvidia-docker，如何拉取镜像并运行，见此[文档](hackathon/setup.md)。

## 大赛介绍
深度学习深刻地改变了计算机应用程序的功能与形态，广泛渗透于我们生活。为了加速深度学习模型的推理，英伟达推出了TensorRT。经过多年的版本迭代，TensorRT在保持极致性能的同时，大大提高了易用性，已经成为GPU上推理计算的必备工具。

随着版本迭代，TensorRT的编程接口在不断更新，编程最佳实践也在不断演化。开发者想知道，为了把我的模型跑在TensorRT上，最省力、最高效的方式是什么？

今天，英伟达联合阿里天池举办TensorRT Hackathon就是为了帮助开发者在编程实践中回答这一问题。英伟达抽调了TensorRT开发团队和相关技术支持团队的工程师组成专家小组，为开发者服务。参赛的开发者将在专家组的指导下在初赛中对给定模型加速；在复赛中自选模型进行加速，并得到专家组一对一指导。

我们希望借助比赛的形式，提高选手开发TensorRT应用的能力，因此重视选手的学习过程以及选手与英伟达专家之间的沟通交流。

## 赛题说明

本赛分初赛和复赛。

### 初赛

初赛是利用 TensorRT 加速 ASR 模型 WeNet（包含 encoder 和 decoder 两个部分），以优化后的运行时间作为主要排名依据。

- 初赛期间我们将建立包含所有选手的技术交流群，供大家研讨用
- 我们专门为此次比赛准备了系列讲座，为了能更顺利地完成比赛，请参赛者观看学习
    - 讲座地址：https://www.bilibili.com/video/BV15Y4y1W73E
    - 配套范例：[cookbook](cookbook)
- 初赛结束时将组织一次讲评，介绍优化该模型的技巧

初赛不提供开发机，参赛选手需要自备带有 GPU 的 Linux / Windows 11 (WSL2) 开发机，并在给定 docker 中用赛方提供的模型文件、开发工具完成模型在 TensorRT 中的构建、精度验证和性能测试，并提交最终代码。

- 初赛使用的镜像：`registry.cn-hangzhou.aliyuncs.com/trt2022/dev` 该镜像基于英伟达官方镜像扩充而来，包含 CUDA 11.6，TensorRT 8.2.2，请使用nvidia-docker拉取并运行它（[示例命令](hackathon/setup.md)）
    - /workspace含有供选手使用的输入文件和测试文件，只读，请勿修改
    - /workspace/encoder.onnx 和 /workspace/decoder.onnx 是在 pyTorch 中训练好的 WeNet 模型的 encoder、decoder 两部分。选手的目标就是把它们转成优化后的TensorRT engine序列化文件（.plan）
    - encoder 相对容易，请优先完成
    - 对于decoder，为了简化起见，请将输入张量 hyps_pad_sos_eos 的末维固定为 64，即在 TensorRT 中构建 engine 时，将该张量形状固为 [-1,10,64]，否则不能完成评测
    - 有能力的选手可以在模型构建成功后尝试 FP16 模式，可取得更好的加速效果

- 代码验证与提交
    - 请保证在 docker 里面能正常运行你的代码，并且无论编译时还是运行时，都不依赖网络下载任何代码或数据，即，你的代码需要是完整的、自包含的（如果确实需要在docker里面新增开发库或软件，请在交流群里反应给赛方）
    - 在代码根目录下，请创建`build.sh`，并保证运行该`build.sh`时，在代码根目录下生成encoder.plan和decoder.plan；如果有plugin，在代码根目录下生成所有 .so
    - 正式提交前，请验证代码已符合要求：
      - 把/target作为代码根目录，把干净代码拷贝过去
      - 运行/workspace/buildFromWorkspace.sh，检查/target下面的.plan和.so是否正常生成
      - 运行/workspace/testEncoderAndDecoder.py，检查TRT engine是否正常运行，并确认在标准输出得到评测表格
    - 验证通过后提交代码：
      - 在[code.aliyun.com](https://code.aliyun.com)上创建代码仓库，设为私有，并把wili-Nvidia加为reporter
        - 注意：不要使用新版的`codeup.aliyun.com`
      - 借助git将自己的代码上传到代码仓库
      - 把仓库的git地址填入天池提交页，正式提交
        - 注意：首次提交代码时，请在天池页面点击“提交结果”->“修改地址”，在弹出的窗口中"git路径"中，请写入可用git clone命令顺利下载代码的URL，比如https://code.aliyun.com/your_name/your_project.git

- 排名依据
  - 以模型优化后在标准评测机器上的GPU运行时间进行排序。如果运行时间非常接近，在测量误差内，则视为成绩相同。
    - 前面所提到的自动生成的“评测表格”会报告你自己的开发机上的GPU运行时间，供你做纵向参考。
    - 实际评测是在标准评测机器上完成的，并不采用你开发机上报告的结果。
  - 对于成绩相同的选手，按提交时间早晚排序。
  - 天池的在线页面有隔一段时间更新的排行榜，它未考虑运行时间的测量误差以及提交时间早晚，其结果仅供参考。初赛结束时赛方将给出最终排行榜。
  - 如果优化后模型的输出结果的精度不合格，则无成绩，不参与排名。前面所提到的自动生成的“评测表格”会报告输出结果的精度是否合格。

### 复赛

参见[复赛说明](hackathon/TRT-Hackathon-2022-final.md)。