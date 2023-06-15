<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-07-04 17:31:00
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-07 15:14:04
-->
<p align="center">
<img src="pic/logo.png" width="600" height="200">
</p>

<div align="center">

# CDSRec: 跨域序列推荐算法工具箱

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/CDSRec)[![](https://img.shields.io/github/license/orion-orion/CDSRec)](https://github.com/orion-orion/CDSRec)[![](https://img.shields.io/github/stars/orion-orion/CDSRec?style=social)](https://github.com/orion-orion/CDSRec)
<br/>
[![](https://img.shields.io/github/directory-file-count/orion-orion/CDSRec)](https://github.com/orion-orion/CDSRec) [![](https://img.shields.io/github/languages/code-size/orion-orion/CDSRec)](https://github.com/orion-orion/CDSRec)

</div>

## 1 简介

[CDSRec](https://github.com/orion-orion/CDSRec)为跨域序列推荐（Cross-Domain Sequential Recommendation）的算法工具箱，旨在提供序列推荐、跨域推荐、跨域序列方法的baseline实现实现。目前本工具箱已包括[TiSASRec](https://dl.acm.org/doi/abs/10.1145/3336191.3371786)<sup>[1]</sup>、[CoNet](https://dl.acm.org/doi/abs/10.1145/3269206.3271684)<sup>[2]</sup>、[PINet](https://dl.acm.org/doi/abs/10.1145/3331184.3331200)<sup>[3]</sup>、[MIFN](https://dl.acm.org/doi/abs/10.1145/3487331)<sup>[4]</sup>这四种方法的实现。

## 2 环境依赖

请注意，该工具箱需要用到 Tensorflow 1.*  ╮（￣▽￣）╭。我的Python版本为 3.8.15，CUDA 版本是 11。因为 Tensorflow 1.15 只支持Python 3.7 和CUDA 10，所以我使用了下列命令以在 CUDA 11 上安装 Tensorflow 1.15:

```bash
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

除了Tensorflow之外，其余环境配置如下：

```text
numpy==1.20.0   
```

## 3 数据集

本项目统一使用[MIFN](https://dl.acm.org/doi/abs/10.1145/3487331)论文中预处理后的Amazon跨域序列推荐数据集（并按照论文[C2DSR](https://dl.acm.org/doi/abs/10.1145/3511808.3557262)<sup>[5]</sup>的建议对其训练/验证/测试集进行了重新划分）。其中包含了Food-Kitchen、Entertainment-Education、Movie-Book这三种不同领域组合所构成数据集。每个数据集都由形如：

*[user_id], A<sub>0</sub> , A<sub>1</sub>, B<sub>0</sub>, A<sub>2</sub>, B<sub>1</sub>, B<sub>2</sub>, B<sub>3</sub>, ... ...*

的序列组成，其中 *[user_id]* 为用户的ID，*A<sub>0</sub> , A<sub>1</sub>,  A<sub>2</sub>* 为用户所交互的领域A的物品，*B<sub>0</sub>, B<sub>1</sub>, B<sub>2</sub>, B<sub>3</sub>* 为用户所交互的领域A物品。具体选择哪些物品作为训练/验证/测试的ground truth跟具体的算法有关，我在`dataset.py`文件中有对数据预处理相关的详细实现，大家可以自行查看。

数据集我已经上传到了[Google drive](https://drive.google.com/drive/folders/1-2RXHsk5dRhpXo6AiLpnDFjkVBTpiquH?usp=sharing)，大家可自行下载并放在项目的`./data`目录下。

## 4 项目目录说明

```bash
CDSRec
├── data                                   存放数据集
├── log                                    存放训练/验证/测试的日志文件
├── tisasrec                               TiSASRec方法的实现
│   ├── __init__.py                        包初始化文件    
│   ├── config.py                          模型配置文件（超参数） 
│   ├── modules.py                         模型子模块
│   ├── tisasrec_model.py                  模型架构
├── conet                                  CoNet方法的实现
│   ├── ...
├── pinet                                  PINet方法的实现
│   ├── ...
├── mifn                                   MIFN方法的实现
│   ├── ...
├── dataset.py                             训练/验证/测试数据集的加载与预处理等操作
├── main.py                                主函数，包括了整个数据pipline
├── train_eval.py                          训练和验证模块
├── data_utils.py                          数据处理相关的工具函数
├── kg_utils.py                            知识图谱相关的工具函数，主要用于MIFN方法
├── .gitignore                             .gitignore文件
├── LICENSE                                LICENSE文件
└── README.md                              README文件
```

## 5 使用方法

运行 `main.py`来加载与预处理数据集，并训练/验证/测试模型:

```bash
python -u main.py \
    --dataset Food-Kitchen \
    --method PINet \
    --epochs 60 \
    --eval_interval 1 \
    --log_dir log \
    --seed 42
```

其中Python解释器的`-u`参数表示标准输出同标准错误一样不通过缓存直接打印到终端。命令行程序的`--dataset`参数用于指定数据集，`--method`参数用于指定客户端的个数，`--epochs`用于指定迭代轮数，`--eval_interval`用于指定模型评估间隔（每隔多少轮进行一次模型验证与测试），`--log_dir`用于指定日志目录，`--seed`用于指定随机数种子。

训练完成后, 你可以在代码运行目录的 `./log`子目录下查看训练/验证/测试的日志情况。

初次训练会对数据集进行预处理，预处理结果存放在`./data/[dataset]/prep_data`目录下（`[dataset]`为数据集名称）形如下面的形式：

```bash
prep_data
├── TiSASRec_train_data.pkl
├── TiSASRec_valid_data.pkl
├── TiSASRec_test_data.pkl
├── ...
```

在之后的训练中，你可以选择添加`--load_prep`参数来加载已经预处理好的数据集。

```bash
python -u main.py \
    --dataset Food-Kitchen \
    --load_prep \
    --method PINet \
    --epochs 60 \
    --eval_interval 1 \
    --log_dir log \
    --seed 42
```

## 参考

[1] Li J, Wang Y, McAuley J. Time interval aware self-attention for sequential recommendation[C]//Proceedings of the 13th international conference on web search and data mining. 2020: 322-330.

[2] Hu G, Zhang Y, Yang Q. Conet: Collaborative cross networks for cross-domain recommendation[C]//Proceedings of the 27th ACM international conference on information and knowledge management. 2018: 667-676.

[3] Ma M, Ren P, Lin Y, et al. π-net: A parallel information-sharing network for shared-account cross-domain sequential recommendations[C]//Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval. 2019: 685-694.

[4] Ma M, Ren P, Chen Z, et al. Mixed information flow for cross-domain sequential recommendations[J]. ACM Transactions on Knowledge Discovery from Data (TKDD), 2022, 16(4): 1-32.

[5] Cao J, Cong X, Sheng J, et al. Contrastive Cross-Domain Sequential Recommendation[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 138-147.
