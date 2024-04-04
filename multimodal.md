###### 多模态模型
* BLIP: 引导语言图像预训练（如上图所示），以实现统一的视觉语言理解和生成。BLIP是一个新的VLP框架，与现有方法相比，它可以实现更广泛的下游任务。它分别从模型和数据角度有两个贡献:

（1） 多模态编码器-解码器混合（MED）：一种用于有效多任务预训练和灵活迁移学习的新模型架构。MED可以作为单模态编码器、基于图像的文本编码器或基于图像的文本解码器工作。该模型与三个视觉语言目标联合预训练：图像文本对比学习、图像文本匹配和图像条件语言建模。

（2） 字幕和过滤（CapFilt）：一种新的数据集增强方法，用于从噪声图像-文本对中学习。作者将预先训练的MED分为两个模块: 一个字幕器，用于生成给定web图像的合成字幕，以及一个过滤器，用于从原始web文本和合成文本中删除嘈杂的字幕。

1. ITC（Image-Text Contrastive Loss）
   cosine_similarity = np.dot(I_e[n,d_e], T_e.T[n, d_e])*np.exp(t)
   label = np.arange(n)
   loss_i = crossentropy_loss(cosine_similarity[n,n], label, axis=0)  
   loss_t = crossentropy_loss(cosine_similarity[n,n], label, axis=1)
   （label的含义是第i行/列的真实label为i）
   ![image](https://github.com/Feve1986/coding/assets/67903547/8d7fac67-fc08-447a-bb7c-427906861db3)

2. ITM（Image-text matching）
   对融合后的特征向量计算交叉熵损失
   ![image](https://github.com/Feve1986/coding/assets/67903547/a8d47d98-1557-4902-a909-5e0c3d8d2d79)

3. LM(Language modeling loss)
   LM时GPT系列的预训练任务。简单来说就是根据前面的词来预测下一个词。与NLP的LM有所不同的是VLP同时将image-embedding引入到上下文信息。最大化自回归序列的似然概率进行训练
![image](https://github.com/Feve1986/coding/assets/67903547/b082dae5-1a14-4972-accc-2d684df260cd)
![image](https://github.com/Feve1986/coding/assets/67903547/b7809fb4-83dd-434c-b4b8-620ee3644e88)

* boostrapping caption（核心）
![image](https://github.com/Feve1986/coding/assets/67903547/4d8fa460-d0ce-4594-ab04-8a159ad4d217)
就是先用人工标注数据预训练好image-grounded text encoder和image-grounded text decoder，然后对来自网络的大量图像文本对进行筛选，筛选的方法是先用decoder生成图像的伪文本标签，然后对encoder判断图像文本对与图像伪文本对是否是相似的，保留图文匹配的图文对，删除不匹配的图文对。

* BLIP2：要从模态对齐、高效训练两个方向对图文多模态预训练任务（vision-and-language pre-training VLP）做出优化。在模态对齐上提出了一个轻量架构QFormer（querying transformer）来建立图像-文本的桥梁。在高效多模态训练上，结合QFormer提出一种二阶段预训练范式来将目前的视觉backbone与LLM模型链接起来。在VQAv2任务上，仅用了 
 倍Flamingo80B的训练数据，却带来8.7%精度提升。

BLIP2的核心是引入了QFormer(Querying Transformer)模块来将对齐图片特征与文本特征。QFormer内部包含两个transformer子模块，其一为image transoformer，其二是text-transformer。image transformer比text-transformer多了一个cross-attention层，这两个transformer共享Self-Attention参数，如下图所示。
![image](https://github.com/Feve1986/coding/assets/67903547/57d85edd-6cdc-4c97-afeb-fafcc848eb30)
这里面有一个需要注意的点：作者没有将image encoder得到的image embedding作为image transformer的输入，而是定义了一个可训练的query作为输入。image embedding用作cross attention层的key， value。

QFormer的目的是，在冻结的视觉模型和大语言模型间进行视觉-语言对齐。Q-Former是一个轻量级的transformer，它使用一个可学习的query向量集，从冻结的视觉模型提取视觉特征。

采取两阶段预训练策略：

阶段一：vision-language表示学习(representation learning)，迫使Q-Former学习和文本最相关的视觉表示。更新的结构是Q-Former。

阶段二：vision-to-language生成式学习(generative learning)，将Q-Former的输出连接到冻结的大语言模型，迫使Q-Former学习到的视觉表示能够为大语言模型所解释。更新的结构是Q-Former和全连接层。

ITC（Image-text Contrastive Learning）：图像的transformer会输出queries那么多个embedding；文本transformer 输入cls token和文本tokens，然后[CLS] token的输出embedding和queries对应的embedding计算相似分数，取最高的作为相似度。这里注意，self-attention时，query和文本token是不交互的！

ITM（Image-Text Matching）：self-attention时，query和文本token是互相交互的。对每个qeury 的输出embedidngs接一个二分类的线性分类器，分图文是否匹配，所有query的分类结果取平均作为最终分类结果。

ITG（Image-grounded Text Generation）：query tokens只跟query tokens交互，文本tokens只跟前面的文本tokens和query tokens交互。生成文本的起始标识token用[DEC]token.。

与常规ITC任务不同的是：单个图片BLIP2产生的image embedding有32个（等于learned query的数量），而text embedding只有1个。BLIP2的操作是，同时计算32个image embedding与text embedding的距离，仅取最近的计算loss。

ITC和ITM主要是为了适应图片分类、图片检索、VQA等理解类任务。ITG主要是为了适应Captioning等生成类任务。

* InstructBLIP：这篇文章在BLIP-2的基础上，在26个数据集上系统研究vision-language的指令集微调（intruction tuning）。并设计了一种instruction-aware的特征提取方式，使得模型能够根据不同instruction来提取图片中特定的表征，以此提升多模态能力。InstructionBLIP是用指令集对训练好的BLIP-2进一步微调。微调的架构如下：![image](https://github.com/Feve1986/coding/assets/67903547/02c13baa-1079-4e43-84a9-21e8d22c697a)

  ![image](https://github.com/Feve1986/coding/assets/67903547/065a136e-5d84-4fdd-90d1-b264054cb68e)

任务类型
![image](https://github.com/Feve1986/coding/assets/67903547/bb31bf59-0d21-4a5a-99b7-03abcf64716b)

问题与解答：
![image](https://github.com/Feve1986/coding/assets/67903547/64103b43-9350-4630-926c-6fd6b365ca11)

###### 增大对比学习的batch
all_gather，[PyTorch多卡分布式训练 | all_gather | 大batch对比学习](https://zhuanlan.zhihu.com/p/615784842)

###### 显存优化
* gradient checkpoint
* 混合精度训练

###### CLIP
![image](https://github.com/Feve1986/coding/assets/67903547/d81498a0-b7d0-4a84-929b-ffee4b530ea5)
```python
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# -------------------------------------------------
# 2. 图像/文字的基本特征过多模态Embedding，提取多模态特征
# 同时对这两个多模态特征做Layer Norm
# -------------------------------------------------
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) # [n, d_i] * [d_i, d_e] = [n, d_e]
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) # [n, d_t] * [d_t, d_e] = [n, d_e]

# -------------------------------------------------
# 3、计算图片-文字向量的余弦相似度
# -------------------------------------------------
logits = np.dot(I_e, T_e.T) * np.exp(t) # [n, n]

# -------------------------------------------------
# 4、计算Loss
# -------------------------------------------------
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

###### AUC
![image](https://github.com/Feve1986/coding/assets/67903547/3f2a5f85-83e0-487b-84b8-6c7a6fc505b5)
![image](https://github.com/Feve1986/coding/assets/67903547/540c421d-50e4-4927-b2ba-0c8a712e8071)
![image](https://github.com/Feve1986/coding/assets/67903547/f855f131-2c23-4ada-823d-816bfa354f84)

###### QKV
们知道K和Q的点乘是为了得到一个attention score 矩阵，用来对V进行提纯。K和Q使用了不同的W_k, W_Q来计算，可以理解为是在不同空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。

###### 面试题集锦
* 词表很大的优化措施：  
   ![image](https://github.com/Feve1986/coding/assets/67903547/07f654ea-8652-4d11-8cca-5821f6c158c1)
   1.表示层分解（Factorized embedding parameterization）
   2.跨层参数共享（Cross-layer parameter sharing）

* CLIP预训练的时候，batch size达到了32768，他用到了哪些trick去提高batch size？ 
  1. 混合精度
  2. gradient checkpoint，时间换空间。只需要存储部分的中间变量，代价是增加了反向传播的计算时间。  
     from torch.utils.checkpoint import checkpoint
  3. 限制每个batch里面的文本最大长度
     原理：利用文本长度分布的长尾特性。比如：你的文本里面99%的文本长度在512个token以内，有一些比较长的文本长度到了2048，但是我们有必要把文本最大长度设置到2048吗？其实很没有必要，512足矣~

使用方法：先统计一下文本长度的分布，根据你的任务和机器资源，设置每个batch_size的最大长度满足99%或者98%（看你的任务），就可以
   4. 增大负样本，这里又有几种方案：
      1. all_gather
      ```python
      with torch.no_grad():
       all_x = [torch.zeros_like(x) for _ in range(world_size)]
       torch.distributed.all_gather(all_x, x)
   all_x[rank] = x
      ```
      ![image](https://github.com/Feve1986/coding/assets/67903547/5ca9a396-fcfb-4908-8dfc-8f920969137f)  
      存在的问题：如果采用的是异步BN，也就是BN层的两个统计参数和两个可学习参数是不同的，那么all_gather后不同GPU上的样本互为负样本，由于不同GPU的样本统计特征不同，使得模型可以根据统计特征中区分出这些为负样本。  
      解决方案：在simCLR[4]中，作者提出的方案是采用所谓的Global BN，其方法就是同样gather不同GPU上的统计参数，然后计算出一个新的统计参数后分发到所有GPU上，此时所有GPU的统计参数都是相同，也就谈不上泄露了。当然你还可以用更简单的方法，比如在[5]中，作者采用Layer Norm取代了Batch Norm。从Fig 4.中可以看出，Layer Norm进行统计参数计算的维度是[Feature, Channel]，而不涉及Batch维度，统计参数不会跨Batch使得统计参数不会泄露样本之间的信息。  
      2. MoCo：维护一个大尺度的负样本队列，并用动量更新的方式去一致更新Query-Key编码器。
      3. Memory Bank


* cv和nlp的经典的对比学习
   1.SimCLR(图像)  
     ![image](https://github.com/Feve1986/coding/assets/67903547/53db6eab-801e-46e3-a944-92d9ae304b33)  
     Loss为InfoNCE Loss，也就是batch内的正样本和负样本计算相似度后做crossentropyloss。
   2.BERT-CT(nlp)，和SimCLR思想基本一致  
   3.SimCSE

![image](https://github.com/Feve1986/coding/assets/67903547/1f957ade-7661-4548-b880-54d8f694ff5c)
