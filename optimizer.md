###### 梯度下降
* 给定待优化的模型参数$\theta\in\mathcal{R}^{d}$和目标函数$J(\theta)$后，
  算法通过沿梯度$\nabla_{\theta} J(\theta)$的相反方向更新$\theta$来最小化$J(\theta)$。
* 1.计算目标函数关于参数的梯度：$g_t = \nabla_{\theta} J(\theta)$
* 2.根据历史梯度计算一阶动量和二阶动量：
  $$m_t = \phi(g_1, g_2, \dots, g_t),v_t = \psi(g_1, g_2, \dots, g_t)$$
* 3.更新模型参数：$\theta_{t+1} = \theta_t - \frac{1}{\sqrt{v_t + \epsilon}} m_t$
###### 各种优化器相同与不同点的内容：
* 朴素SGD(随机梯度下降)：没有动量的概念，$\theta_{t+1} = \theta_t - \eta g_t$，如何合理设置学习率是朴素SGD的难点。
> 做法：随机选择一部分样本，并根据其梯度来更新参数。
> 在遇到沟壑时容易陷入震荡，引入动量加速SGD在正确方向的下降并抑制震荡。
* SGD-M：引入了动量，加速 SGD 在正确方向的下降并抑制震荡。一阶动量的更新公式为：$m_t = \gamma m_{t-1} + \eta g_t.$
  SGD-M 在原步长之上，增加了与上一时刻步长相关的$\gamma m_{t-1}$，$\gamma$通常取 0.9 左右。
  这意味着参数更新方向不仅由当前的梯度决定，也与此前累积的下降方向有关。这使得参数中那些梯度方向变化不大的维度可以加速更新，并减少梯度方向变化较大的维度上的更新幅度。
  由此产生了加速收敛和减小震荡的效果。
* AdaGrad：自适应梯度算法。SGD以相同的学习率去更新$\theta$的各个分量，而不同参数的更新频率往往有所区别。
  AdaGrad引入了二阶动量：$$v_t = \text{diag}\left(\sum_{i=1}^{t} g_{i,1}^2, \sum_{i=1}^{t}g_{i,2}^2,\dots, \sum_{i=1}^{t} g_{i,d}^2\right).$$此时可以认为学习率等效于$\frac{\eta}{\sqrt{v_t + \epsilon}}.$
* AdaDelta/RMSProp：自动调整学习率。在AdaGrad中，随着训练周期的增长，学习率降低的很快。  
  RMS在计算二阶动量时不累积全部历史梯度，而只关注最近某一时间窗口内的下降梯度。其二阶动量采用指数平均移动公式计算。
>SGD-M在SGD的基础上考虑了指数移动平均的一阶动量；AdaDelta/RMSProp在SGD的基础上考虑了指数移动平均的二阶动量。
* Adam是RMSProp和Momentum的结合，保证迭代比较平稳。
  和 RMSprop 对二阶动量使用指数移动平均类似，Adam 中对一阶动量也是用指数移动平均计算。
  $$m_t = \eta[\beta_1 m_{t-1} + (1 - \beta_1)g_t],v_t = \beta_2 v_{t-1} + (1 - \beta_2) \cdot \text{diag}(g_t^2).$$
  注意到，在迭代初始阶段，$m_t$和$v_t$有一个向初值的偏移（过多的偏向了 0）。  
  因此，可以对一阶和二阶动量做偏置校正 (bias correction):
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
* AdamW：在Adam基础上针对L2正则项进行解耦合操作，添加下图中的绿色项而非粉色项：![image](https://github.com/Feve1986/coding/assets/67903547/2c57c554-089d-4233-a5f2-73fafc1ffe70)
  此时参数更新公式为：$w_{t+1} = (1 - \alpha_t \lambda)w_t - \alpha_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}},$其中$\lambda$为权重衰减参数，通常取0.01。


