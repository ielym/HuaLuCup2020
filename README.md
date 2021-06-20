# HuaLuCup2020
2020中国华录杯数据湖算法大赛 吸烟打电话赛道 rank4

## 赛题描述
* 比赛连接 [2020中国华录杯·数据湖算法大赛—定向算法赛（吸烟打电话检测）](https://dev.ehualu.com/dev/home/competition/competitionDetail?competitionId=3)

本赛题基于这些数据，初赛需要识别<b>吸烟</b>，<b>打电话</b>， <b>正常</b>三个类别，复赛额外添加了<b>同时吸烟打电话</b>的类别，难点在于数据集中人物清晰度有限，而且人物的像素大小不一，。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/1.jpg" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/2.jpg" height="200" /> 
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/6.jpg" height="200" /> 
</p>

## 比赛流程和任务
比赛分为初赛和复赛两个阶段。但由于初赛前期数据质量较低，主办方中间更换过一次数据集，因此根据实际算法设计，实际打榜过程可以分为 <b>初赛阶段一</b>，<b>初赛阶段二</b>
和 <b>复赛阶段</b>。

下面将会分别按照这三个阶段，主要介绍算法设计和数据处理过程

## 初赛阶段一
### 数据分析
初赛更换高质量数据集之前，主要难点在于训练集和测试集边缘分布差距较大，且训练集中存在大量噪声图像。其中训练集中<b>正常</b>类别和测试集的分布
比较接近，都是监控场景下的图像。而训练集中的<b>吸烟</b>和<b>打电话</b>类别多为网络图像或影视截图，测试集中仍然以监控场景下的图像为主。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/4.jpg" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/5.jpg" height="200" /> 
</p>