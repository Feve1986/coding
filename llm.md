###### 怎么解决模型的输出结尾出现重复字符的问题：
> 模型的训练数据一般是正向问题，尽量不要出现否定的语句。

###### 为什么大模型难以使用长文本
简而言之，self-attention 使同一上下文的每一个token都能够观察到上下文得任意位置的 token（decoder 的 self-attention 是每个token只能观察到上下文任意历史位置的token），这一结构特点使得Transformer 较之CNN、RNN等模型结构理论上显著提升了长距离依赖的捕捉能力，无数实验也证实了这种结构可以提升最终的效果。付出的代价就是，与此同时计算复杂度也增长为平方级。计算时间和计算资源的制约是 n 无法迅速增大的直接因素，也就是在平方级复杂度制约下，模型上下文难以随心所欲地增长。
![image](https://github.com/Feve1986/coding/assets/67903547/bfcd1e9a-6385-4f05-8fc4-8bf96c1d5945)

###### 长上下文的解决方案
1.借助模型外部工具辅助处理长文本或者利用外部记忆存储过长的上下文向量，可以称为外部召回的方案；
2.利用模型优化的一般方法：包括量化（Quantization）、剪枝（Pruning）、蒸馏（Distillation）、参数共享（Weight Sharing）、矩阵分解（Factorization）
3.优化Attention的计算（Efficient Transformers）：
  Transformer-XL(相对位置编码), Sparse Patterns，Low-Rank Transformation，Memory / Downsampling
4. RopE的位置编码扩展
5. 

###### 流水线并行

###### 上下文长度越长越好吗
* 上下文过长会导致注意力分散。
  
###### kimi
通过创新的网络结构和工程优化，克服大模型在长文本处理上的挑战，不依赖于滑动窗口、降采样、小模型等性能性能损失较大的方案，实现了支持20w字输入的200b参数大模型kimi。研发更长上下文长度的模型，以覆盖更多场景并提高应用效果。
kimi的优势：
* 长文本处理能力：支持20w字输入。
* 无损记忆能力：在处理长文本信息时保持信息的完整性和连贯性。
* 高效的长程注意力机制：不依赖于滑动窗口、降采样、小模型等简化处理方法。
* 多语言能力
  
###### Baichuan-192k
上下文窗口长度支持192k token，约等于35w汉字。实现技术：
* 算法层面：提出一种基于RoPE和ALiBi的动态位置编码的外推方案，对不同分辨率的ALiBi_mask进行不同程度的Attention-mask动态内插，
  在保证分辨率的同时增强模型对长序列依赖的建模能力。
* 工程层面：在自主开发的分布式训练框架上，整合优化技术如张量并行、流水并行、序列并行、重计算和Offload功能
* Bfloat16混合精度训练；采用了NormHead，对输出embedding进行归一化处理

|fp16|bf16|fp32|
|----|----|----|
|范围问题|精度问题|显存问题|
|1(sign)+5(exp)+10(frac)|1+8+7|1+8+23|

由于自注意力机制中使用了大量的 exp 运算，因此很多情况下，更大的范围比精度重要很多（能够有效防止上下溢出）。
![image](https://github.com/Feve1986/coding/assets/67903547/fe30e220-5db6-4c81-a474-45d814acca13)

###### InfLLM（支持1024k输入）
1. 在滑动窗口的基础上，加入远距离的上下文记忆模块。  
2. 将历史上下文切分成语义块，构成上下文记忆模块中的记忆单元。每个记忆单元通过其在之前注意力计算中的注意力分数确定代表性Token，作为记忆单元的表示。从而避免上下文中的噪音干扰，并降低记忆查询复杂度

###### Gemini
Gemini 支持高达 1000万 token 的超长上下文和强大的多模态能力，这意味着利用 Gemini 能够与整本书籍、庞大的文档集、数百个文件组成的数十万行代码库、完整电影、整个播客系列等进行交互。

###### Claude：先监督微调 后RAIHF
RAIHF：通过AI排序而非人工排序数据集训练出来的偏好模型PM的指引下迭代模型策略

###### Llama2
相比于 Llama 1 ，Llama 2 的训练数据多了 40%，上下文长度也翻倍，并采用了分组查询注意力机制。具体来说，Llama 2预训练模型是在2 万亿的 token上训练的，精调 Chat 模型是在100 万人类标记数据上训练的。
![image](https://github.com/Feve1986/coding/assets/67903547/c51025fa-670b-425a-a2c1-e1ec3f283c69)
![image](https://github.com/Feve1986/coding/assets/67903547/fe751fe3-3b1d-4c97-9f1d-76b277e9bdcc)

1. 训练数据Token数量从1.4T->2T
2. 序列长度从2K->4K
3. 在SFT过程中，LLAMA2强调数据质量的重要性，通过2W的高质量指令数据，激发模型的指令遵循能力。
4. 在RLHF过程中，LLAMA2做了较多工作，对RLHF过程作出了进一步的解释。自建了100W的Reward数据集，训练了两个独立的Reword Model。
5. PreNorm

###### ChatGLM：
1. 基于 FlashAttention 技术，将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K
2. 基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用
3. ChatGLM2-6B 使用了 GLM 的混合目标函数
4. PostNorm




###### DPO
RLHF的替代算法：直接偏好优化(Direct Preference Optimization，简称DPO)。DPO通过简单的分类目标直接优化最满足偏好的策略，而没有明确的奖励函数或RL。与RLHF一样，DPO依赖于理论偏好模型，衡量给定的奖励函数与经验偏好数据的一致性。

DPO利用从奖励函数到最优策略的解析映射，这使我们能够将奖励函数上的偏好损失函数转换为策略上的损失函数。具体做法是给定人类对模型响应的偏好数据集，DPO使用简单的二元交叉熵目标优化策略，而无需在训练期间明确学习奖励函数或从策略中采样。

###### Llama为什么使用RoPE
1. 序列长度灵活性、长度外推
2. 随着相对距离的增加而衰减的tokens间的依赖性，
3. 以及，为线性自注意力，配备（配装的）的相对位置编码的能力。
> RoPE对q和k做
![image](https://github.com/Feve1986/coding/assets/67903547/51d8ab07-fe1b-4664-9b41-60ee19fe4ca5)

长度外推的做法
![image](https://github.com/Feve1986/coding/assets/67903547/ad00967f-a786-4da9-bac4-e699586c5f55)


###### 子词分词器
* BPE：

1. BPE首先使用一个普通的分词器将语料切分成词，分词器可以选择前面提到的基于空格或基于规则的分词器。

2. 分完词后就得到了一个包含所有唯一单词的词表，并统计词表中每个单词出现的频次。

3. 创建一个包含了步骤2中的词表中所有符号的基础词汇表。

4. 从基础词汇表中选择两个符号根据合并规则形成一个新的符号，并更新基础词表。（BPE根据新符号出现的频率来合并的）

5. 重复第4步，直到基础词表的大小达到想要的大小。(词表的大小是一个预先定义的超参数)

* WordPiece:
1. 首先使用一个普通的分词器将语料切分成词，分词器可以选择前面提到的基于空格或基于规则的分词器。

2. 分完词后就得到了一个包含所有唯一单词的词表。

3. 创建一个包含了步骤2中的词表中所有符号的基础词汇表。

4. 从所有可能的组合中选择加入词表后能最大程度地增加语言模型在训练数据上的似然概率的组合

5. 重复第4步，直到基础词表的大小达到想要的大小。(词表的大小是一个预先定义的超参数)
![image](https://github.com/Feve1986/coding/assets/67903547/9384a79b-df58-467f-aff5-b20a9d5047de)

###### 激活函数
GELU是ReLU的平滑版本。

RELU的优缺点：
![image](https://github.com/Feve1986/coding/assets/67903547/954028f2-8ddf-48a7-b59a-7661c7f25c1c)

![image](https://github.com/Feve1986/coding/assets/67903547/51aecfc2-fd5d-44ab-824c-74a0cef3e54e)

###### jieba分词（中文分词框架）
![image](https://github.com/Feve1986/coding/assets/67903547/558e6836-5440-4ecd-8588-dea1bd1418c4)

###### 语言模型评价指标
困惑度（概率的高低，概率越高困惑度越低，不是最重要的指标）

###### 参数估计
* 最大似然估计：就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值。极大似然估计中采样需满足一个重要的假设，就是所有的采样都是独立同分布的。
  ![image](https://github.com/Feve1986/coding/assets/67903547/da9c52d4-28c7-426f-8f4c-f7933322ff1b)

 [DPO: Direct Preference Optimization 论文解读及代码实践] (https://zhuanlan.zhihu.com/p/642569664)

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

###### RAG
RAG 已经被证明是一种解决大模型幻觉的有效方法，如何进一步提升 RAG 的实战效果？

1.提升长上下文的理解能力

由于嵌入模型通常较小，上下文窗口有限，传统的 RAG 通常依赖分块对数据进行切分。这导致了上下文信息的丢失，例如一些代词的信息无法连贯地理解。

举例来说明，在某个对话中，提到 “Bill 周日去了埃菲尔铁塔，之后又跟朋友一起去了卢浮宫”。当我们进行传统的提问，例如询问“Bill周日下午去了哪里？”时，由于上下文信息被分割成多个分块，可能会导致搜索到的信息仅包含了Bill周日去了埃菲尔铁塔，从而形成了错误的结论。这种情况下，由于上下文被切分，系统无法正确理解代词“去了哪里”的指代对象，从而导致了错误的结果。

近期，基于大型模型实现的嵌入逐渐成为主流。在 Huggingface MTEB LeaderBoard 中，效果最好的嵌入基本上都是由大型模型所霸榜。这一趋势的一个副产品是嵌入的上下文窗口也逐渐提升，例如 SRF-Embedding-Mistral 和 GritLM7B 已经支持 32k 的长上下文，这意味着嵌入本身处理长上下文的能力也得到了大幅提升。

最近发布的 BGE Landmark embedding 的论文也阐述了一种利用长上下文解决信息不完整检索的方法。通过引入无分块的检索方法，Landmark embedding 能够更好地保证上下文的连贯性，并通过在训练时引入位置感知函数来有限感知连续信息段中最后一个句子，保证嵌入依然具备与 Sentence Embedding 相近的细节。这种方法大幅提升了长上下文 RAG 的精度。

2.利用多路召回提升搜索质量

为了提升 RAG 的回复质量，关键在于能够检索到高质量的内容。数据清理、结构化信息提取以及多路混合查询，都是提高搜索质量的有效手段。最新的研究表明，相比稠密向量模型，使用如 Splade 这类稀疏向量模型，在域外知识搜索性能，关键词感知能力以及可解释方面表现更佳。最近开源的 BGE_M3 模型能够在同一模型中生成稀疏、稠密以及类似 Colbert 的 Token 多向量，通过不同类型的向量多路召回并结合大型模型进行排名，可以显著提高检索效果。这种混合查询的方法也被向量数据库厂商广泛接受，最近发布的 Milvus 2.4 版本也支持了稠密和稀疏向量的混合查询。

3.使用复杂策略提升RAG能力

开发大模型应用不仅面临算法挑战，还涉及复杂的工程问题。这要求开发者具备深入的算法知识及复杂系统设计和工程实践的能力。采用复杂策略，如查询改写、意图识别和实体检测，不仅提升了准确性，也显著加快了处理速度。即使是先进的 Gemini 1.5 模型，在进行 Google 的 MMLU 基准测试时也需调用 32 次才能达到 90.0% 的准确率，显示出采用复杂工程策略以提升性能的必要性。通过使用向量数据库和 RAG，采取空间换时间的策略，使 RAG 系统能更有效利用大型语言模型（LLM）的能力。这不仅限于生成答案，还包括分类、提取结构化数据、处理复杂 PDF 文档等任务，增强了 RAG 系统的多功能性，使其能适应更广泛的应用场景。
![image](https://github.com/Feve1986/coding/assets/67903547/34cc63ef-3ad4-4463-b7ce-0a48d7afc7f3)
[RAG 修炼手册｜RAG 敲响丧钟？大模型长上下文是否意味着向量检索不再重要](https://segmentfault.com/a/1190000044755011)

###### Agent
* 什么是Agent: Model+Planning+Memory+Tools. 有工具，有记忆，能规划。


###### BERT
* 手推BERT的参数量：Embedding+Encoder+Pooling：
1. Embedding：Embedding(V+512+2)\*d+Norm(2\*d)=(V+516)*d
2. Encoder：每个Block：Multi-Head Attention(4\*d\*d+4\*d)+Add&Norm(2\*d)+Feed Forward(d\*(4\*d)+4*\d+(4\*d)*d+d)+Add&Norm(2\*d)=12\*(12\*d**2+13\*d)
3. Pooling：(d\*d+d)=(d**2+d)
* BERT的训练任务：Masked LM（完形填空）和 Next Sentence Prediction，是下一个句子则为1，不是则为0，做二分类。
* 激活函数为Gelu（一般只在FFN加激活函数）
  ![image](https://github.com/Feve1986/coding/assets/67903547/7b8049c5-fd59-4899-8825-6dab5cb769c2)

###### Ernie与Bert的不同
* Ernie是百度发布的一个预训练模型，创新点在于
  1. Mask的方式有所升级，在BERT的训练中，是以字符为单位进行训练的，而ERNIE则将MASK分为了3个级别：字符级、实体级、短语级，个人理解比较像分词。
  2. Dialog embedding：对话训练数据

###### 提示工程
上下文理解，角色扮演，Coc，CoT思考链

###### 模型联接
* 内联-MoE：Expert，Route，Gating Model, 常见实现有GPT-4
* 外联-Agent：Brain（处理和记忆），Perception（多模感知），Action（输出和控制）。常见实现有Langchain

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

* RLHF  
1. RLHF可以分为三个步骤：  
   1. 预训练一个语言模型 (LM)
   2. 聚合问答数据并训练一个奖励模型 (Reward Model，RM)
   3. 用强化学习 (RL) 方式微调 LM。方案是近端策略优化（Proximal Policy Optimization，PPO） 

2. 奖励函数：
奖励模型（RM 模型）将 SFT 模型最后一层的 softmax 去掉，即最后一层不用 softmax，改成一个线性层。RM 模型的输入是问题和答案，输出是一个标量即分数。由于模型太大不够稳定，损失值很难收敛且小模型成本较低，因此，RM 模型采用参数量为 6B 的模型，而不使用 175B 的模型。

> 奖励模型的损失函数采用 Pairwise Ranking Loss
![image](https://github.com/Feve1986/coding/assets/67903547/529a7b7a-330c-4a43-bf0e-d185d6b27a58)
![image](https://github.com/Feve1986/coding/assets/67903547/6764d0c9-645a-44ef-9f54-ffa5c47b2148)

> PPO 算法确定的奖励函数具体计算如下：将提示 输入初始 LM 和当前微调的 LM，分别得到了输出文本 ，将来自当前策略的文本传递给 RM 得到一个标量的奖励 。将两个模型的生成文本进行比较计算差异的惩罚项，在来自 OpenAI、Anthropic 和 DeepMind 的多篇论文中设计为输出词分布序列之间的 Kullback–Leibler (KL) 散度的缩放，即 。这一项被用于惩罚 RL 策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。此外，OpenAI 在 InstructGPT 上实验了在 PPO 添加新的预训练梯度，可以预见到奖励函数的公式会随着 RLHF 研究的进展而继续进化。

> 最后根据 PPO 算法，我们按当前批次数据的奖励指标进行优化 (来自 PPO 算法 on-policy 的特性) 。PPO 算法是一种信赖域优化 (Trust Region Optimization，TRO) 算法，它使用梯度约束确保更新步骤不会破坏学习过程的稳定性。DeepMind 对 Gopher 使用了类似的奖励设置，但是使用 A2C (synchronous advantage actor-critic) 算法来优化梯度。最后根据 PPO 算法，我们按当前批次数据的奖励指标进行优化 (来自 PPO 算法 on-policy 的特性) 。PPO 算法是一种信赖域优化 (Trust Region Optimization，TRO) 算法，它使用梯度约束确保更新步骤不会破坏学习过程的稳定性。DeepMind 对 Gopher 使用了类似的奖励设置，但是使用 A2C (synchronous advantage actor-critic) 算法来优化梯度。

将初始语言模型的微调任务建模为强化学习（RL）问题，需要定义策略（policy）、动作空间（action space）和奖励函数（reward function）等基本要素。策略就是基于该语言模型，接收 prompt 作为输入，然后输出一系列文本（或文本的概率分布）；而动作空间就是词表所有 token 在所有输出位置的排列组合；观察空间则是可能的输入 token 序列（即 prompt），为词表所有 token 在所有输入位置的排列组合；而奖励函数则是上一阶段训好的 RM 模型，配合一些策略层面的约束进行的奖励计算。该阶段流程如下图所示：

RL 模型训练的损失函数公式如下：
![image](https://github.com/Feve1986/coding/assets/67903547/a5d0af43-1bf6-454b-be53-ae063d39e73f)
RL 模型的优化目标是使得损失函数越大越好，损失函数可以分为三个部分，打分部分、KL 散度部分以及预训练部分。
![image](https://github.com/Feve1986/coding/assets/67903547/dbe092f7-4005-4b12-8775-1d806711f741)

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

LoRA：LoRA添加的位置一般是q和v。

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
