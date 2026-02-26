# minimind-study
## RMSNorm

梯度与x本身值有关，x过大或过小，容易引起梯度爆炸或梯度消失，需要将其归一化变成均值为0、方差为1

RMSNorm比LayerNorm少计算了均值，加快计算速度，而且效果不错
$$
y_i=\frac{x_i}{\sqrt{\frac{1}{n}\sum(x_i)^2}+\epsilon}*\gamma
$$
其中$\gamma$是一个可学习的参数

## RoPE

RoPE通过一种旋转变换，**将位置信息直接融入到q和k的表示中**。RoPE将位置信息的引入方式从原来的“对于输入token的加法操作”变成了“对于q和k的旋转操作”，并且**以绝对位置编码的形式实现了相对位置编码**。

假设现有m和n位置的token，对应$q_n$和$k_m$，对应位置的角频率为$\theta=\omega * pos$

定义二维旋转：
$$

\text{RoPE}(\mathbf{x}, \theta) = 
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\cdot \mathbf{x}
$$
旋转后的q和k相乘：
$$
\tilde{\mathbf{q}}^\top \cdot \tilde{\mathbf{k}} = \mathbf{q}^\top R(-\theta_q) R(\theta_k) \cdot \mathbf{k}
= \mathbf{q}^\top R(\theta_k - \theta_q) \cdot \mathbf{k}
$$
注意力变成了与 $(n - m)$​相关的点积结果。

RoPE的基础频率$freqs_i=\frac{1}{base^{2i/dim}}$

## GQA

