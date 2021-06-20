# HuaLuCup2020
2020中国华录杯数据湖算法大赛 吸烟打电话赛道 rank4

## 赛题描述
* 比赛连接 [2020中国华录杯·数据湖算法大赛—定向算法赛（吸烟打电话检测）](https://dev.ehualu.com/dev/home/competition/competitionDetail?competitionId=3)
行为规范，即指某些特定的场景会对人物的行为做出特定的限制，比如加油站禁止吸烟，驾驶员禁止打电话，博物馆禁止拍照等。随着计算机人工智能的发展，这些禁止行为或者不文明行为都可通过基于视频监控的行为检测算法进行监控、发现以及适当时给与警告。

<center class="half">
<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg" width="30%"/><img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg" width="30%"/>
</center>

![](https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg)| ![](https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg)

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg" width="30%" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/ME-DIS-3.jpg" width="30%" /> 
</p>

本赛题基于这些数据，初赛需要识别<b>吸烟</b>，<b>打电话</b>， <b>正常</b>三个类别，复赛额外添加了同时吸烟打电话的<b>吸烟打电话</b>类别，难点在于数据集中人物清晰度有限，而且人物的像素大小不一，。

## 比赛流程和任务
比赛分为初赛和复赛两个阶段。但由于初赛前期数据质量较低，主办方中间更换过一次数据集，因此根据本人实际算法设计，实际打榜过程可以分为 <b>初赛阶段一</b>，<b>初赛阶段二</b>
和 <b>复赛阶段</b>。

下面将会分别按照这三个阶段，主要介绍算法设计和数据处理过程

## 初赛阶段一
### 数据分析
初赛更换高质量数据集之前，主要难点在于训练集和测试集边缘分布差距较大，且训练集中存在大量噪声图像。其中<b>正常</b>