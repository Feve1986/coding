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


```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)
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

###### KV Cache
* KV cache参数量计算
  KV cache需要的参数量为：2\*2\*(s+n)\*h\*l*b=4blh(s+n). 其中第一个2表示K/V cache，第二个2表示FP16占两个bytes，l表示层数，h表示隐藏层维度，s和n分别为输入序列和输出序列的长度。  
kv cache可以分为两个阶段，第一阶段为prompt输入，第二阶段为token by token的内容输出。
![image](https://github.com/Feve1986/coding/assets/67903547/be29f6a7-1cbf-4136-84fc-c68c6c2a2d4e)
![image](https://github.com/Feve1986/coding/assets/67903547/07aabae3-c089-4c2f-93ed-81aeecce59cd)

随着batch size和长度的增大，kv cache占用的显存开销快速增大，甚至会超过模型本身。

而LLM的窗口长度也在不断增大，因此就出现一组主要矛盾，即：对不断增长的LLM的窗口长度的需要与有限的GPU显存之间的矛盾。因此优化KV cache就显得非常必要。

* KV cache优化的典型方法
1. MQA（Multi Query Attention，多查询注意力）是多头注意力的一种变体。主要区别在于，在MQA中不同的注意力头共享一个K和V的集合，每个头只单独保留了一份查询参数。因此K和V的矩阵仅有一份，这大幅度减少了显存占用，使其更高效。由于MQA改变了注意力机制的结构，因此模型通常需要从训练开始就支持MQA。也可以通过对已经训练好的模型进行微调来添加多查询注意力支持，仅需要5%的原始训练数据量就可以达到不错的效果。
![image](https://github.com/Feve1986/coding/assets/67903547/0683aa59-33f9-463d-a501-151964db1717)
2. 窗口优化：KV cache的作用是计算注意力，当推理时的文本长度T大于训练时的最大长度L时，一个自然的想法就是滑动窗口。
3. 量化与稀疏
4. 存储与计算优化
5. 

[参考文章](https://grs.zju.edu.cn/cas/login?service=http%3A%2F%2Fgrs.zju.edu.cn%2Fallogene%2Fpage%2Fhome.htm)
[对 Transformer 显存占用的理论分析](https://zhuanlan.zhihu.com/p/462443052)
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
###### 激活函数
Gelu：xP(X<=x), 其中X为服从标准正态分布的随机变量。

###### BERT
* 手推BERT的参数量：Embedding+Encoder+Pooling：
1. Embedding：Embedding(V+512+2)\*d+Norm(2\*d)=(V+516)*d
2. Encoder：每个Block：Multi-Head Attention(4\*d\*d+4\*d)+Add&Norm(2\*d)+Feed Forward(d\*(4\*d)+4*\d+(4\*d)*d+d)+Add&Norm(2\*d)。=12\*(12\*d**2+13\*d)
3. Pooling：(d\*d+d)=(d**2+d)
* BERT的训练任务：Masked LM（完形填空）和 Next Sentence Prediction，是下一个句子则为1，不是则为0，做二分类。
* 激活函数为Gelu（一般只在FFN加激活函数）
  
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
* Ernie是百度发布的一个预训练模型，创新点在于
  1. Mask的方式有所升级，在BERT的训练中，是以字符为单位进行训练的，而ERNIE则将MASK分为了3个级别：字符级、实体级、短语级，个人理解比较像分词。
  2. Dialog embedding：对话训练数据

###### Agent是什么

###### 显存占用分析
* 训练过程
在训练神经网络的过程中，占用显存的大头主要分为四部分：模型参数、前向计算过程中产生的中间激活、后向传递计算得到的梯度、优化器状态。

在一次训练迭代中，每个可训练模型参数都会对应1个梯度，并对应2个优化器状态（Adam优化器梯度的一阶动量和二阶动量）。
float16数据类型的元素占2个bytes，float32数据类型的元素占4个bytes。在混合精度训练中，会使用float16的模型参数进行前向传递和后向传递，计算得到float16的梯度；在优化器更新模型参数时，会使用float32的优化器状态、float32的梯度、float32的模型参数来更新模型参数。
![image](https://github.com/Feve1986/coding/assets/67903547/6eea10a4-7d9a-4657-b78c-ac0b742d42b5)

除了模型参数、梯度、优化器状态外，占用显存的大头就是前向传递过程中计算得到的中间激活值了，需要保存中间激活以便在后向传递计算梯度时使用。这里的激活（activations）指的是：前向传递过程中计算得到的，并在后向传递过程中需要用到的所有张量。这里的激活不包含模型参数和优化器状态，但包含了dropout操作需要用到的mask矩阵。

* 推理过程
  在神经网络的推理阶段，没有优化器状态和梯度，也不需要保存中间激活。少了梯度、优化器状态、中间激活，模型推理阶段占用的显存要远小于训练阶段。模型推理阶段，占用显存的大头主要是模型参数，如果使用float16来进行推理，推理阶段模型参数占用的显存大概是2*参数量bytes。如果使用KV cache来加速推理过程，KV cache也需要占用显存，KV cache占用的显存下文会详细介绍。此外，输入数据也需要放到GPU上，还有一些中间结果（推理过程中的中间结果用完会尽快释放掉），不过这部分占用的显存是很小的，可以忽略。

###### FlashAttention
FlashAttention用于加速self-attention运算，其关键在于有效的硬件使用。

FlashAttention的优化思路：1.在不访问整个输入的情况下计算softmax 2.不为反向传播存储大的中间attention矩阵。

为此，FlashAttention提出了两种方法来解决上述问题，tiling和recomputation。

tiling的主要思想是分割输入，将它们从慢速HBM加载到快速SRAM，然后计算这些快的attention输出。

内存高效：传统的注意力机制（例如普通注意力）存在二次内存复杂度 (O(N²))，其中 N 是序列长度。另一方面，Flash Attention 将内存复杂度降低到线性 (O(N))。这种优化是通过有效利用硬件内存层次结构并最大限度地减少不必要的数据传输来实现的。

###### logistic损失函数（交叉熵）


###### MHA
* 作用
  1. 每个注意力头使用不同的线性变换，这意味着它们可以从输入序列的不同子空间中学习不同的特征关联。
  2. 并行运算，提升性能。
* scale/sqrt(d):不做scale难以收敛，容易出现梯度消失问题。

```python
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights

if __name__ == '__main__':
    heads = 2
    batch_size = 1
    seq_len=3
    d_model = 10
    multiheadattention = Multiheadattention(heads,d_model,dropout)
    x = torch.randn(batch_size,seq_len,input_dim)
    print(x)  # batch_size,seq_len,embeding
    attention = multiheadattention(x)
    print(attention)   
```

###### 模型微调


###### 知识图谱
![image](https://github.com/Feve1986/coding/assets/67903547/c753e19e-133e-4878-82e5-cb4fe5637945)
1. 向量数据库-RAG-外挂知识
2. 把过去的知识图谱整理为训练语料进行训练
3. 用于构造prompt

###### Chatgpt训练的三个阶段 
1. 预训练，主要针对补全能力（给问题添加上下文，添加后续问题，给出实际答案）
2. 监督微调（问答、摘要、翻译）：优化预训练模型，使其生成用户所期望的回答。
3. 基于强化学习的人类指令微调
   1. 训练一个作为评分函数的奖励模型。
   2. 优化LLM以生成能够在奖励模型中获得高分的回答。
![image](https://github.com/Feve1986/coding/assets/67903547/3b8de1aa-3e4f-4b62-9c41-73144584f8cd)

PPO(Proximal Policy Optimization): 在这一过程中，提示会从一个分布中随机选择，例如，我们可以在客户提示中进行随机选择。每个提示被依次输入至LLM模型中，得到一个回答，并通过RM给予回答一个相应评分。约束条件：这一阶段得到的模型不应与SFT阶段和原始预训练模型偏离太远（在下面的目标函数中以KL散度项进行数学表示）

幻觉问题：外挂知识，线性探测。

###### 大模型
Llama，Llama2，ChatGLM，Chatpgt系列，Kimi，Baichuan

* ChatGLM：
  1. 基于 FlashAttention 技术，将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K
  2. 基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用
  3. ChatGLM2-6B 使用了 GLM 的混合目标函数
  4. PostNorm

* Llama2:
  1. 训练数据Token数量从1.4T->2T
  2. 序列长度从2K->4K
  3. 在SFT过程中，LLAMA2强调数据质量的重要性，通过2W的高质量指令数据，激发模型的指令遵循能力。
  4. 在RLHF过程中，LLAMA2做了较多工作，对RLHF过程作出了进一步的解释。自建了100W的Reward数据集，训练了两个独立的Reword Model。
  5. PreNorm

###### Llama
* tokenization：BPE(Byte Pair Encoding)算法

  核心就是根据出现频率不断合并直到减少到词表大小或概率增量低于某一阈值。

*  LlamaMLP 中一共有 3 个 Linear 层
```python
class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # config 中 hidden_act = 'silu'
        # 'silu' 和 'swish' 对应的激活函数均为：SiLUActivation 
        # https://github.com/huggingface/transformers/blob/717dadc6f36be9f50abc66adfd918f9b0e6e3502/src/transformers/activations.py#L229
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # 对应上述公式的 SwiGLU
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

* 位置编码：RoPE
  不同于原始 Transformers 论文中，将 pos embedding 和 token embedding 进行相加，RoPE 是将位置编码和 query （或者 key） 进行相乘。
* RMS Norm 的全称为 Root Mean Square layer normalization。与 layer Norm 相比，RMS Norm的主要区别在于去掉了减去均值的部分。RMS Norm 的作者认为这种模式在简化了Layer Norm 的计算，可以在减少约 7%∼64% 的计算时间。

* 使用SwiGLU替代了ReLU作为激活函数。FFN层包括两个线性变换，中间插入一个非线性激活函数。最初的Transformer架构采用了ReLU激活函数。

  
###### 反向传播
* crossentropy反向传播的梯度计算：[手推公式之“交叉熵”梯度](https://zhuanlan.zhihu.com/p/518044910)

###### 怎么解决灾难性遗忘
灾难性遗忘现象是在连续学习多个任务的过程中，学习新知识的过程会迅速破坏之前获得的信息，而导致模型性能在旧任务中急剧下降。尤其是在微调大语言模型时，微调后导致模型在旧任务中急剧下降。
造成灾难性遗忘的一个主要原因是，传统模型假设数据分布是固定或平稳的， 训练样本是独立同分布的，所以模型可以一遍又一遍地看到所有任务相同的数据， 但当数据变为连续的数据流时，训练数据的分布就是非平稳的，模型从非平稳的数据分布中持续不断地获取知识时，新知识会干扰旧知识，从而导致模型性能的快速下降。

为了克服灾难性遗忘，持续学习（一直给模型喂数据，一直让模型学）是一种能够缓解深度学习模型灾难性遗忘的机器学习方法，包括正则化方法、记忆回放方法和参数孤立等方法，为了扩展模型的适应能力，让模型能够在不同时刻学习不同任务的知识，即模型学习到的数据分布，持续学习算法必须在保留旧知识与学习新知识之间取得平衡。
还可以用预训练的权重衰减，学习率衰减、对抗性微调和参数高效微调等方法解决灾难性遗忘。

###### PEFT
p tuning v2 和 prompt tuning  lora的区别，各自的优缺点 

Prompt tuning：分为hard和soft两种形式，hard是输入prompt是不可导的，soft是将一个可训练张量与输入文本的embeddings拼接起来，这个可训练张量可以通过反向传播来优化

Prefix tuning：与prompt tuning相似，主要区别如下：prefix tuning将prefix参数（可训练张量）添加到所有的transformer层，而prompt tuning只将可训练矩阵添加到输入embedding。具体地，prefix tuning会将prefix张量作为past_key_value添加到所有的transformer层，并用一个独立的FFN来编码和优化prefix参数。

Adapter：把额外的可训练参数添加到每个transformer层。与prefix tuning不同之处是：prefix tuning是把prefix添加到输入embedding；而adapter在两个层之间插入了adapter 层，adapter层由全连接层+激活函数+全连接层组成。

LLAMA-Adapter：只给L个深层transformer层添加了可学习的adapter，且这个adapter是一个self-attention层

LoRA

###### 大模型训练框架
DeepSpeed, Megatron,Zero

###### 为什么会出现 LLMs 复读机问题？
数据偏差：大型语言模型通常是通过预训练阶段使用大规模无标签数据进行训练的。如果训练数据中存在大量的重复文本或者某些特定的句子或短语出现频率较高，模型在生成文本时可能会倾向于复制这些常见的模式。

训练目标的限制：大型语言模型的训练通常是基于自监督学习的方法，通过预测下一个词或掩盖词来学习语言模型。这样的训练目标可能使得模型更倾向于生成与输入相似的文本，导致复读机问题的出现。

缺乏多样性的训练数据：虽然大型语言模型可以处理大规模的数据，但如果训练数据中缺乏多样性的语言表达和语境，模型可能无法学习到足够的多样性和创造性，导致复读机问题的出现。

模型结构和参数设置：大型语言模型的结构和参数设置也可能对复读机问题产生影响。例如，模型的注意力机制和生成策略可能导致模型更倾向于复制输入的文本。

为了解决复读机问题，可以采取以下策略：
多样性训练数据：在训练阶段，尽量使用多样性的语料库来训练模型，避免数据偏差和重复文本的问题。

引入噪声：在生成文本时，可以引入一些随机性或噪声，例如通过采样不同的词或短语，或者引入随机的变换操作，以增加生成文本的多样性。

温度参数调整：温度参数是用来控制生成文本的多样性的一个参数。通过调整温度参数的值，可以控制生成文本的独创性和多样性，从而减少复读机问题的出现。

后处理和过滤：对生成的文本进行后处理和过滤，去除重复的句子或短语，以提高生成文本的质量和多样性。

添加惩罚项对重复进行限制


