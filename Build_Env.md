### 1. Install CUDA 10.2 & CUDNN 8.2.0
##### step - 1. cuda10.2
访问该网址，并根据自己的平台选择并下载
* [cuda_10.2.89](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64)

##### step - 2. cudnn 8.2.0
1. 访问该网址，登录NVIDIA帐户并根据平台选择下载
* [cuDNN v8.2.0 for CUDA 10.2](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse820-102)
2. 下载后解压, 解压后的文件夹结构应当为
```
cuda
├─bin
├─include
└─lib
    └─x64
```
将`bin`,`include`,`lib`三个文件夹复制到cuda安装路径 \
例如：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2`

> cuda10.2 & cudnn8.2 for windows10 直接下载: \
> [cuda_10.2.89_441.22_win10.exe](https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe) \
> [cudnn-10.2-windows10-x64-v8.2.0.53.zip](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.0.53/10.2_04222021/cudnn-10.2-windows10-x64-v8.2.0.53.zip) \
> 参考资料: [win10安装cuda10.2和CUDNN8.2.2](https://blog.csdn.net/mao_hui_fei/article/details/119564307)

### 2. Install miniconda
##### step - 1. Download Anaconda3
[Anaconda3-2021.11-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-2021.11-Windows-x86_64.exe)
##### step - 2. Install it
> 参考资料: [怎么安装Anaconda3](https://zhuanlan.zhihu.com/p/75717350)

## 3. 创建环境
##### step - 1. create conda env
```shell
conda create -n Jetracer python=3.8
conda activate Jetracer
```

##### step - 2. Install PyTorch1.9.0
```shell
# cuda10.2
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```

##### step - 3. Install jupyter clickable image widget
> 如`jupyter labextension install js` 报错 可考虑跳过step-3安装 \
> 使用`check_and_modify_un_change_mode.ipynb`替换`check_and_modify.ipynb` \
> 或将`check_and_modify.ipynb`运行于Jetson Nano
1. install jupyter
```shell
conda install -c conda-forge jupyterlab
```

2. install Node.js from conda
```shell
conda install -c conda-forge nodejs==16.14.2
```

3. build jupyter-widgets
```shell
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

4. clone `jupyter_clickable_image_widget` and install it
```shell
git clone https://github.com/jaybdub/jupyter_clickable_image_widget
cd jupyter_clickable_image_widget
pip install -e .
jupyter labextension install js
```

##### step - 4. Install others
```shell
pip install -r requirements.txt
```

---
##### 可选安装 TensorRT & torch2trt
> TensorRT和torch2trt仅用于非Jetson平台使用`utils/multi_model_inference.py`和`utils/trt_speed.py`进行加速后模型结果查看
1. TensorRT 8.2.4
    1. 下载并解压缩：\
       [tensorrt-8.2.4.2.windows10.x86_64.cuda-10.2.cudnn8.2.zip](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.4/zip/tensorrt-8.2.4.2.windows10.x86_64.cuda-10.2.cudnn8.2.zip)
    2. 将解压缩后的文件放置到`<安装文件夹>`
       > `<安装文件夹>`指用户自行决定的安装文件夹
    3. 将`<安装文件夹>/lib`添加到系统变量
    4. 切换到希望安装`tensorrt`的python环境
        ```shell
        pip install {tensorrt}/python/tensorrt-8.2.4.2-cp38-none-win_amd64.whl
        pip install {tensorrt}/uff/uff-0.6.9-py2.py3-none-any.whl
        pip install {tensorrt}/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
        pip install {tensorrt}/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
        ```
2. torch2trt
    ```shell
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    python setup.py instal
    ```