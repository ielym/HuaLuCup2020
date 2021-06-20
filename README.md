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
* 初赛更换高质量数据集之前，主要难点在于训练集和测试集边缘分布差距较大，且训练集中存在大量噪声图像。其中训练集中<b>正常</b>类别和测试集的分布
比较接近，都是监控场景下的图像。而训练集中的<b>吸烟</b>和<b>打电话</b>类别多为网络图像或影视截图，测试集中仍然以监控场景下的图像为主。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/4.jpg" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/5.jpg" height="200" /> 
</p>

* 为了缓解背景差距较大（监控场景和网络图像/影视截图），使用了MixUP数据增强策略来解决这个问题。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/7.png" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/8.png" height="200" /> 
</p>

* 此外，对于样本类别分布和图像高宽比也进行了统计。针对标签分布不一致的问题，使用了翻转，旋转，高斯噪声等方式进行了扩充；针对图像尺度差距
较大，以及高宽比差距较大的问题，分别使用了多尺度输入图像和Padding的方式进行训练。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/9.jpg" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/10.jpg" height="200" /> 
</p>

### 模型设计
* 首先尝试了直接使用<b>ResNext101_32x8d</b>网络提取特征+<b>Softmax</b>分类器进行多分类的方式，但线上评分只有0.5左右。
* 结合上述数据分析过程，认为对于数据分布差异较大的情况下，不适合使用<b>Softmax</b>进行分类，转而使用<b>SVM</b>分类器增大类间距，并使用两个二分类器 :
<b>是否是"正常类别"</b>，<b>"吸烟"还是"打电话"</b>，这种方案获得了线上0.8的评分，截止官方更换数据集，排名在第5名。

<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/11.png" width="100%" />

## 初赛阶段二
### 数据分析
* 根据参赛选手的反馈，官方在初赛中更换了训练集和测试集，数据质量有了很大的提升。主要体现在训练集和测试集的边缘分布更加相似，监控场景下的图像
和网络图像/影视截图的标签分布更加接近。

### 模型设计
* 针对数据质量较高的情况下，且分类任务较为简单（三分类）。因此就直接使用<b>ResNext101_32x8d</b>提取特征+<b>Softmax</b>分类的方案。
* 部分图像中，<b>吸烟</b>，<b>打电话</b>的区域较小，无法有效的从少量的像素中提取出有效的特征进行分类，因此使用了CBAM注意力机制来增强这些
特征的表达。
* 初赛最终网络结构如下：

<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/12.png" width="100%" />

## 复赛阶段
### 数据分析
* 复赛阶段添加了<b>同时吸烟打电话</b>的类别。此外，还增加了<b>图像中出现香烟和手机，但没有在人手里的干扰图像</b>，以及人在图像中所占比例非常小的情况。

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/15.jpg" height="200" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/13.jpg" height="200" /> 
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/14.jpg" height="200" /> 
</p>

* 为了更好的提升模型的泛化能力，从数据角度出发，我们更加细致的把数据分为了7种场景，并针对各个场景进行数据扩充（翻转等，爬取Ins图像）。对于爬取
的图像，我们使用<b>伪标签</b>进行了标注。7种场景，和扩充后的数据分布如下图所示：

<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/16.png" width="100%" />

<p float="left">
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/17.png" height="180" />
  <img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/18.png" height="180" /> 
</p>

* 训练集中 <b>干扰图像</b> 的数量较少，因此我们爬取并裁剪了手机，香烟的图像，并在行人重识别数据集上进行了复制粘贴来进行扩充。

<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/19.png" height="200" /> 
<img src="https://github.com/ielym/HuaLuCup2020/blob/main/datas/train/20.png" height="200" /> 

### 模型设计