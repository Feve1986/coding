###### 交叉熵损失函数：
* 信息量I=-log(p)  
* 熵：所有信息量的期望。熵的公式：  
![image](https://github.com/Feve1986/coding/assets/67903547/490beaa0-f2d3-4f39-9b56-eed2ac75eef6)
* 交叉熵又称KL散度，最小化交叉熵等价于最小化KL散度。其公式为：
![image](https://github.com/Feve1986/coding/assets/67903547/d6dd5e00-7f78-43b3-9ef4-98a3f79c45a4)
* 下面是一个多类别的交叉熵损失计算例子：
![image](https://github.com/Feve1986/coding/assets/67903547/25df6009-074a-4eeb-af9d-441142bef277)
* 对于二分类问题，交叉熵损失的公式可以简化为如下形式：
![image](https://github.com/Feve1986/coding/assets/67903547/bc552784-e774-449d-b51b-fe094ea01dbf)
* 对于一张图片有多个类别的情形：
![image](https://github.com/Feve1986/coding/assets/67903547/e48e952f-c734-4d53-920b-8049e736bff6)

> 另一个角度的意义：对于同一个随机变量 x，如果有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异。
![image](https://github.com/Feve1986/coding/assets/67903547/64c65453-87d0-41e1-b9dd-d363a28c37c0)
* KL散度公式经过变形后有如下形式：![image](https://github.com/Feve1986/coding/assets/67903547/c0f8631d-b741-4e5e-b4f8-887cbb768e3c)
> 因此损失函数其实是在衡量两个分布之间的差异。

* 和最大似然函数是等价的：
![image](https://github.com/Feve1986/coding/assets/67903547/80bc4a72-2b8a-4a41-beca-5ecffba45d93)


###### MSE损失函数：
* ![image](https://github.com/Feve1986/coding/assets/67903547/8521be24-5cb7-4c19-80fe-2148929532d4)
不使用MSE损失的原因：
1. MSE 的损失小于交叉熵的损失，导致对分类错误的点的惩罚不够：
![image](https://github.com/Feve1986/coding/assets/67903547/9d822ee6-2919-4a61-afc8-a290d76440f3)
2. MSE 的梯度消失效应较大：
![image](https://github.com/Feve1986/coding/assets/67903547/ad1297ba-0998-477d-b318-de0ff5fd2810)
3. 损失函数的凸性（使用MSE可能会陷入局部最优）
![image](https://github.com/Feve1986/coding/assets/67903547/a6f813ea-45fb-454d-8170-011a45bebe76)

> 随机投硬币
> sigmoid：![image](https://github.com/Feve1986/coding/assets/67903547/da7a029e-64a4-4386-8d3d-45b622b84845)


###### FocalLoss
* focalloss：![image](https://github.com/Feve1986/coding/assets/67903547/09cb67a6-3680-4acc-b642-2066273d443c)

  相当于增加了分类不准确样本在损失函数中的权重。

  focal loss相当于增加了难分样本在损失函数的权重，使得损失函数倾向于难分的样本，有助于提高难分样本的准确度。focal loss与交叉熵的对比，
