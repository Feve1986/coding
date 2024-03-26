###### 图像分割传统方法
* 基于阈值的图像分割
* 基于区域的图像分割：区域生长
* 基于边缘检测的图像分割：Canny算子
###### 视频分割方法
* 基于时域的视频对象分割方法：
  > 基本思想：时域分割主要时利用视频相邻图像之间的连续性和相关性进行分割。  
  > 具体做法：一种是通过当前帧和背景帧相减来获得差分图像，另外一种是利用两帧之间或者多帧之前的差来获得差分图像。
* 基于运动的视频对象分割方法
  > 基本思想：主要是基于光流场等方法进行运动参数估计，求出符合运动模型的像素区域，进而合并区域构成运动对象进行视频分割。  
  > 具体做法：首先求出光流场并进行参数估计，接着求出符合运动模型的像素区域，最后合并区域构成运动对象进行视频分割。
  
###### 强化学习
* 强化学习与传统的优化决策算法相比，具有以下几个优势：
  > 基于经验学习：传统的优化决策算法通常需要一个数学模型来描述系统，然后使用优化技术求解最优解。而强化学习不需要事先了解系统模型，而是通过与环境的交互来学习系统的行为。
  > 能够处理复杂的环境：传统的优化决策算法通常需要假设环境是静态的，并且是可预测的，但在现实中，很多环境都是非线性、非静态、非确定性的，这使得传统方法难以应对。强化学习则可以在不知道环境模型的情况下，学习如何在这样的环境中做出最佳决策。
  > 能够处理长期回报问题：强化学习可以处理长期回报问题，也就是在某一时间点做出的决策可能在未来的时间点才会得到回报的情况。传统的优化决策算法通常是基于短期的目标来做出决策的，而强化学习则能够考虑长期的影响。
  > 能够自适应地学习：强化学习算法具有自适应性，可以在不断与环境交互的过程中不断地改进自己的决策策略，逐渐逼近最优策略。传统的优化决策算法则通常是一次性求解出最优解，不能适应环境变化。

###### Norm
* Layer Norm:LN特别适合处理变长数据，因为是对channel维度做操作(这里指NLP中的hidden维度)，和句子长度和batch大小无关
* Batch Norm
* PowerNorm: Rethinking Batch Normalization in Transformers
> 那对于处理CV data的VIT，可不可以用BN呢？Leveraging Batch Normalization for Vision Transformers里面就说了：其实可以的，
  但是直接把VIT中的LN替换成BN，容易训练不收敛，原因是FFN没有被Normalized，所以还要在FFN block里面的两层之间插一个BN层。（可以加速20% VIT的训练）
> Transformer中BN表现不太好的原因可能在于CV和NLP数据特性的不同，对于NLP数据，前向和反向传播中，batch统计量及其梯度都不太稳定。
> 对于NLP data来说，batch上去做归一化是没啥意义的，因为不同句子的同一位置的分布大概率是不同的。
![image](https://github.com/Feve1986/coding/assets/67903547/f886fc95-8aed-4a3e-840c-fd9ebb8c3794)

```python
class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__() #确保父类
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
	
  def forward(self, x):
  	# 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
  	# 相当于变成[bsz*max_len, hidden_dim], 然后再转回来, 保持是三维
  	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
  	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
          # 注意这里也在最后一个维度发生了广播
  	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

###### 数据增强方式
* 数据增强的作用
1. 避免过拟合。当数据集具有某种明显的特征，例如数据集中图片基本在同一个场景中拍摄，使用Cutout方法和风格迁移变化等相关方法可避免模型学到跟目标无关的信息。

2. 提升模型鲁棒性，降低模型对图像的敏感度。当训练数据都属于比较理想的状态，碰到一些特殊情况，如遮挡，亮度，模糊等情况容易识别错误，对训练数据加上噪声，掩码等方法可提升模型鲁棒性。

3. 增加训练数据，提高模型泛化能力。

4. 避免样本不均衡。在工业缺陷检测方面，医疗疾病识别方面，容易出现正负样本极度不平衡的情况，通过对少样本进行一些数据增强方法，降低样本不均衡比例。

* 数据增强的分类
根据数据增强方式，可分为两类：在线增强和离线增强。这两者的区别在于离线增强是在训练前对数据集进行处理，往往能得到多倍的数据集，在线增强是在训练时对加载数据进行预处理，不改变训练数据的数量。

离线增强一般用于小型数据集，在训练数据不足时使用，在线增强一般用于大型数据集。

* 数据增强的类型
比较常用的几何变换方法主要有：翻转，旋转，裁剪，缩放，平移，抖动。值得注意的是，在某些具体的任务中，当使用这些方法时需要主要标签数据的变化，如目标检测中若使用翻转，则需要将gt框进行相应的调整。

比较常用的像素变换方法有：加椒盐噪声，高斯噪声，进行高斯模糊，调整HSV对比度，调节亮度，饱和度，直方图均衡化，调整白平衡等。

这些常用方法都比较简单，这里不多赘述。

Cutout：对一张图像随机选取一个小正方形区域，在这个区域的像素值设置为0或其它统一的值。注：存在50%的概率不对图像使用Cutout。

Mixup：将在数据集中随机选择两张图片按照一定比例融合，包括标签值

Hide-and-Seek：将图片划分为S x S的网格，每个网格按一定的概率（0.5）进行掩码。其中不可避免地会完全掩码掉一个完整的小目标。

CutMix：选择一个小区域，进行掩码，但掩码的方式却是将另一张图片的该区域覆盖到这里

GridMask：使用的方式是排列的正方形区域来进行掩码。

FenceMask：![image](https://github.com/Feve1986/coding/assets/67903547/be2f36c5-7e1e-4dd2-832c-f6d62e1e679b)

KeepAugment：分析出最不重要的区域，选择这个区域进行Cutout，或者分析出最重要区域进行CutMix。

多样本数据增强：

1.Mosaic：使用四张图片拼接成一张图片

2.SamplePairing：从训练集中随机选择两张图片，经过几何变化的增强方法后，逐像素取平均值的方式合成新的样本

生成网络：基于GAN来实现风格迁移

###### Loss
* focalloss：![image](https://github.com/Feve1986/coding/assets/67903547/09cb67a6-3680-4acc-b642-2066273d443c)

  相当于增加了分类不准确样本在损失函数中的权重。

  focal loss相当于增加了难分样本在损失函数的权重，使得损失函数倾向于难分的样本，有助于提高难分样本的准确度。focal loss与交叉熵的对比，

###### 微调方案
LoRA、Adapter、前缀微调
LoRA添加的位置一般是k和v。

###### ROC曲线的绘制
* TPR，FPR：TPR（recall）的意义是所有真实类别为1的样本中，预测类别为1的比例。FPR的意义是所有真实类别为0的样本中，预测类别为1的比例。
![image](https://github.com/Feve1986/coding/assets/67903547/61a8aabc-5709-4183-ba5e-b5df722edc30)
* F1 score： ![image](https://github.com/Feve1986/coding/assets/67903547/c22a416c-e0b1-4625-9d82-9db16754e780)
* ![image](https://github.com/Feve1986/coding/assets/67903547/9ac19c50-91be-4082-ba28-e0dde49e4cb3)

###### 卷积层
* 卷积的参数量为输入维度\*输出维度\*卷积核尺寸

###### kv cache
* KV cache参数量计算
  KV cache需要的参数量为：2\*2\*(s+n)\*h\*l*b=4blh(s+n). 其中第一个2表示K/V cache，第二个2表示FP16占两个bytes，l表示层数，h表示隐藏层维度，s和n分别为输入序列和输出序列的长度。  
kv cache可以分为两个阶段，第一阶段为prompt输入，第二阶段为token by token的内容输出。
![image](https://github.com/Feve1986/coding/assets/67903547/07aabae3-c089-4c2f-93ed-81aeecce59cd)

随着batch size和长度的增大，kv cache占用的显存开销快速增大，甚至会超过模型本身。

而LLM的窗口长度也在不断增大，因此就出现一组主要矛盾，即：对不断增长的LLM的窗口长度的需要与有限的GPU显存之间的矛盾。因此优化KV cache就显得非常必要。

* KV cache优化的典型方法
  MQA（Multi Query Attention，多查询注意力）是多头注意力的一种变体。主要区别在于，在MQA中不同的注意力头共享一个K和V的集合，每个头只单独保留了一份查询参数。因此K和V的矩阵仅有一份，这大幅度减少了显存占用，使其更高效。由于MQA改变了注意力机制的结构，因此模型通常需要从训练开始就支持MQA。也可以通过对已经训练好的模型进行微调来添加多查询注意力支持，仅需要5%的原始训练数据量就可以达到不错的效果。

[参考文章](https://grs.zju.edu.cn/cas/login?service=http%3A%2F%2Fgrs.zju.edu.cn%2Fallogene%2Fpage%2Fhome.htm)
###### 激活函数
Gelu：xP(X<=x), 其中X为服从标准正态分布的随机变量。

###### BERT
* 手推BERT的参数量：Embedding+Encoder+Pooling：
1. Embedding：Embedding(V+512+2)\*d+Norm(2\*d)=(V+516)*d
2. Encoder：每个Block：Multi-Head Attention(4\*d\*d+4\*d)+Add&Norm(2\*d)+Feed Forward(d\*(4\*d)+4*\d+(4\*d)*d+d)+Add&Norm(2\*d)。=12\*(12\*d**2+13\*d)
3. Pooling：(d\*d+d)=(d**2+d)

###### CLIP
* 结合了检索模型和生成模型
* 预训练：从互联网收集的4亿图像文本对

###### GPT-4的语言能力
通过Text Encoder来编码句子，通过Image Encoder来编码图像，然后计算文本嵌入和图像嵌入的相似度。
视频生成的关键：生成的视频的关键帧之间信息不会丢失。首先需要理解图像。
跨帧之间的信息传输。
Sora：数据预处理：可变持续时间、分辨率、横纵比。统一的视觉表示，视频压缩网络，时空潜伏斑块，Diffusion Transformer

###### 提示工程
上下文理解，角色扮演，Coc，CoT思考链

###### 模型联接
* 内联-MoE：Expert，Route，Gating Model, 常见实现有GPT-4
* 外联-Agent：Brain（处理和记忆），Perception（多模感知），Action（输出和控制）。常见实现有Langchain

###### Ernie与Bert的不同

###### Agent是什么