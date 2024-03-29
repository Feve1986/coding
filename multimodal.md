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
就是先用人工标注数据预训练好image-grounded text encoder和image-grounded text decoder，然后对来自网络的大量图像文本对进行筛选，筛选的方法是先用decoder生成图像的伪文本标签，然后对encoder判断图像文本对与图像伪文本对是否是相似的，如果相似则同时保留两者，否则同时抛弃两者。

* BLIP2：要从模态对齐、高效训练两个方向对图文多模态预训练任务（vision-and-language pre-training VLP）做出优化。在模态对齐上提出了一个轻量架构QFormer（querying transformer）来建立图像-文本的桥梁。在高效多模态训练上，结合QFormer提出一种二阶段预训练范式来将目前的视觉backbone与LLM模型链接起来。在VQAv2任务上，仅用了 
 倍Flamingo80B的训练数据，却带来8.7%精度提升。

BLIP2的核心是引入了QFormer(Querying Transformer)模块来将对齐图片特征与文本特征。QFormer内部包含两个transformer子模块，其一为image transoformer，其二是text-transformer。image transformer比text-transformer多了一个cross-attention层，这两个transformer共享Self-Attention参数，如下图所示。

![image](https://github.com/Feve1986/coding/assets/67903547/57d85edd-6cdc-4c97-afeb-fafcc848eb30)
这里面有一个需要注意的点：作者没有将image encoder得到的image embedding作为image transformer的输入，而是定义了一个可训练的query作为输入。image embedding用作cross attention层的key， value。
