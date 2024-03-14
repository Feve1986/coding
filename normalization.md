>在把数据喂给机器学习模型之前，“白化（whitening）”是一个重要的数据预处理步骤。白化一般包含两个目的：  
>（1）去除特征之间的相关性 —> 独立；  
>（2）使得所有特征具有相同的均值和方差 —> 同分布。  
>白化最典型的方法就是PCA，可以参考阅读 PCAWhitening。  

###### Iternal Covariant Shift(ICS)
* ICS会导致什么问题？
  简而言之，每个神经元的输入数据不再是“独立同分布”。  
  其一，上层参数需要不断适应新的输入数据分布，降低学习速度。  
  其二，下层输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。  
  其三，每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

###### 归一化的类型
* Batch Normalization：![image](https://github.com/Feve1986/coding/assets/67903547/1c0f7034-10fc-47d3-97a4-509ae1fe463c)
  将 normalize 后的数据再扩展和平移，是为了让神经网络自己去学着使用和修改这个扩展参数$\gamma$, 和平移参数$\beta$,
  这样神经网络就能自己慢慢琢磨出前面的 normalization 操作到底有没有起到优化的作用,
  如果没有起到作用, 就使用$\gamma$和$\beta$来抵消一些 normalization 的操作。
* Layer Normalization：该层的所有神经元进行Normalization
* Group Normalization：沿channel方向按照num_groups分成num_groups份，对每份求均值和方差然后进行计算。
* Weight Normalization
![image](https://github.com/Feve1986/coding/assets/67903547/8e257f3f-0fdb-446c-85cd-46e759505e8c)

>除了充分利用底层学习的能力，另一方面的重要意义在于保证获得非线性的表达能力。
>Sigmoid 等激活函数在神经网络中有着重要作用，通过区分饱和区和非饱和区，使得神经网络的数据变换具有了非线性计算能力。
>而第一步的规范化会将几乎所有数据映射到激活函数的非饱和区（线性区），仅利用到了线性变化能力，
>从而降低了神经网络的表达能力。而进行再变换，则可以将数据从线性区变换到非线性区，恢复模型的表达能力。

> 模型常用的剪枝方法：根据BN层的$\gamma$参数（方差），方差越大则特征越明显。剪枝时去除方差小的层，保留方差大的层。


###### 激活函数

###### 两种不进行反向传播机制的区别
* model.eval: 依然计算梯度，但是不反传；dropout层保留概率为1；batchnorm层使用全局的mean和var
* with torch.no_grad: 不计算梯度
