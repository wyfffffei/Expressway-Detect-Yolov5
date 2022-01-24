# YOLOV5_Note

> @ wyfffffei



## Install CUDA & CUDNN

> 确保电脑有GPU，并且C盘控件足够（CUDA安装解压默认在C盘）

### Pytorch官网查看匹配的版本

> https://pytorch.org/get-started/locally/

- 本文使用的是CUDA11.3



### NVIDIA官网下载对应版本的CUDA和CUDNN

> https://developer.nvidia.com/cuda-11.3.0-download-archive
>
> https://developer.nvidia.com/rdp/cudnn-archive

- CUDA11.3对应的CUDNN版本为：CUDNN v8.2.0

- 分别下载（cuDNN下载需要注册）



### 安装CUDA和CUDNN

- 安装CUDA前需要先下载 Visual Studio 2019 
- 打开安装包，默认选项到底
- 解压CUDNN文件，将所有文件复制到CUDA主目录（默认：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3）
- 添加用户环境变量（默认安装路径）：
  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin
  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp

```cmd
C:\> nvidia-smi
Tue Jan 25 02:08:44 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 511.23       Driver Version: 511.23       CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
...
# 验证安装（重启？）
```



### 配置Pytorch环境

- 回到Pytorch官网：<https://pytorch.org/get-started/locally/>
- 根据本地环境，选择版本，操作系统，包管理工具等
- 本文使用 Stable (1.10.1) + Windows + Pip + Python + CUDA 11.3

```cmd
# 官网自动生成
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```python
import torch

print(torch.cuda.is_available())
```

返回 True 就 OK 了，返回 FALSE 建议把电脑砸了

