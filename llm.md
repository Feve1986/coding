###### 怎么解决模型的输出结尾出现重复字符的问题：
> 模型的训练数据一般是正向问题，尽量不要出现否定的语句。
###### 为什么大模型难以使用长文本
简而言之，self-attention 使同一上下文的每一个token都能够观察到上下文得任意位置的 token（decoder 的 self-attention 是每个token只能观察到上下文任意历史位置的token），这一结构特点使得Transformer 较之CNN、RNN等模型结构理论上显著提升了长距离依赖的捕捉能力，无数实验也证实了这种结构可以提升最终的效果。付出的代价就是，与此同时计算复杂度也增长为平方级。计算时间和计算资源的制约是 n 无法迅速增大的直接因素，也就是在平方级复杂度制约下，模型上下文难以随心所欲地增长。
![image](https://github.com/Feve1986/coding/assets/67903547/bfcd1e9a-6385-4f05-8fc4-8bf96c1d5945)
###### 解决方案
* 1.借助模型外部工具辅助处理长文本或者利用外部记忆存储过长的上下文向量，可以称为外部召回的方案；
* 2.利用模型优化的一般方法：包括量化（Quantization）、剪枝（Pruning）、蒸馏（Distillation）、参数共享（Weight Sharing）、矩阵分解（Factorization）
* 3.优化Attention的计算（Efficient Transformers）：
  Transformer-XL(相对位置编码), Sparse Patterns，Low-Rank Transformation，Memory / Downsampling
###### 上下文长度越长越好吗
* 上下文过长会导致注意力分散。
