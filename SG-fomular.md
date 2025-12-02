# **SG-formula**

Let $R_E$ denote the radius of the Earth. The orbital radii of the lower-layer computing satellites and the upper-layer remote sensing satellites are defined as $R_1 = R_E + h_1$ and $R_2 = R_E + h_2$, respectively, where $h_1$ and $h_2$ are their orbital altitudes ($h_2 > h_1$).

![image-20251130140504220](C:\Users\charl\AppData\Roaming\Typora\typora-user-images\image-20251130140504220.png)

对于下面公式给出示意图
$$
\cos \theta = \frac{R_1^2 + R_2^2 - R_{\text{search}}^2}{2 R_1 R_2}
    \label{eq:cos_theta}
$$
球冠的公式如下：
$$
S = 2 \pi R_1^2 (1 - \cos \theta)
    \label{eq:spherical_area}
$$
将公式(1)带入(2)可得
$$
S = \frac{\pi R_1}{R_2} \left[ R_{\text{search}}^2 - (R_2 - R_1)^2 \right]
    \label{eq:final_area}
$$
### 计算任务分析：

在 M/M/1 稳定系统 ($\rho < 1$) 中，系统服务排队时间 $T$ 的**累积分布函数 (CDF)** $F_T(t)$ 定义为 $P(T \le t)$，即顾客在系统中停留时间不超过 $t$ 的概率，其公式为：

$$
F_T(t) = 1 - \rho e^{-(\mu - \lambda) t}, \quad t \ge 0
$$
其中：

- $t$ 是时间变量， $t \ge 0$。
- $\rho=\frac{\lambda}{\mu}$
- $\mu (1-\rho)$ 是有效服务率，它等于 $\mu - \lambda$。

### MPPP分析：

泊松点过程稀疏化定理（PPP Thinning Theorem）是随机几何中的一个核心结论。该定理指出，如果对一个原始的泊松点过程（PPP，强度为 $\lambda(x)$）中的每一个点，都以一个独立于其他点的概率 $p(x)$ 将其保留下来，那么所有被保留下来的点集合，仍然构成一个新的泊松点过程。当卫星满足等待服务时间小于$\frac{\bar{T}_{pass}}2$，卫星才会参与协助计算。
$$
p_{\text{available}} = 1 - \rho e^{-(\mu - \lambda)\frac{\bar{T}_{pass}}2}
$$
对于MPPP，其卫星密度等于
$$
\lambda_{\text{eff}} = \lambda_{\text{real}} \times p_{\text{available}}
$$
Given the satellite density $\rho$ of the lower layer, the expected number of available computing satellites $m$ within the coverage area is given by:
$$
N_{eff} = \left\lfloor \lambda_{eff} \cdot S \right\rfloor = \left\lfloor \lambda_{eff} \cdot \frac{\pi R_1}{R_2} \left( R_{\text{search}}^2 - (h_2 - h_1)^2 \right) \right\rfloor 
$$

### **Amdahl 定律加速比公式：**  

$$
S_{Amdahl} = \frac{1}{f + \frac{1-f}{N_{eff}}}
$$

其中对于参数的说明：

- $f$: 不可并行部分的比例（例如，预处理、后处理、通信开销）。
- $N$: 参与协同计算的卫星（GPU）数量。

### **卫星选择策略：**

因此协助计算后整个系统可以被看作是一个新的MM1系统，$\lambda$依旧不变，但是$\mu_{eff}=\mu \cdot S_{Amdahl}$

因此系统总逗留时间小于$\tau$的概率可以表示为以下公式：
$$
F_{W}(\tau) = 
\begin{cases}
    0 & \text{if } \tau < 0 \\
    1 - e^{-(\mu_{eff} - \lambda)\tau} & \text{if } \tau \ge 0
\end{cases}
$$

### 反解出N：

我们令这个概率等于给定的任务完成概率 
$$
P_{task} = 1 - e^{-(\mu_{eff} - \lambda)\tau}
$$
从上式中解出指数项：
$$
e^{-(\mu_{eff} - \lambda)\tau} = 1 - P_{task}
$$
对两边取自然对数 $\ln$即可解出 $(\mu_{eff} - \lambda)$，因此，**有效服务率 $\mu_{eff}$** 的表达式为：
$$
\mu_{eff} - \lambda = - \frac{1}{\tau} \ln(1 - P_{task})
$$
将参数带入进行化简可得：
$$
S_{Amdahl} = \frac{\lambda}{\mu} - \frac{1}{\mu \tau} \ln(1 - P_{task})
$$
得到卫星数量 $N$ 的闭式表达式
$$
N \ge \frac{(1-f) \left( \frac{\lambda}{\mu} - \frac{1}{\mu \tau} \ln(1 - P_{task}) \right)}{1 - f \left( \frac{\lambda}{\mu} - \frac{1}{\mu \tau} \ln(1 - P_{task}) \right)}
$$
由于 $N$ 必须是**整数**，因此您需要的最小卫星数量 $N_{min}$ 为：
$$
N_{min} = \lceil N \rceil = \left\lceil \frac{(1-f) \left( \frac{\lambda}{\mu} - \frac{1}{\mu \tau} \ln(1 - P_{task}) \right)}{1 - f \left( \frac{\lambda}{\mu} - \frac{1}{\mu \tau} \ln(1 - P_{task}) \right)} \right\rceil
$$
其中：

- $N$: 参与协同计算的卫星数量。
- $P_{task}$: 期望的任务完成概率（即系统总逗留时间小于 $\tau$ 的概率）。
- $\tau$: 给定的时间限制。
- $\lambda$: 任务到达率。
- $\mu$: 单个卫星（系统）的服务率。
- $f$: 任务中不可并行部分的比例。

### 搜索半径分析：

假设 $N_{eff}$ 是由排队论分析得出的卫星数量 $N$，则搜索半径 $R_{\text{search}}$ 的闭式表达式为：
$$
R_{\text{search}} = \sqrt{\frac{N_{min} \cdot R_2}{\lambda_{eff} \cdot \pi R_1} + (h_2 - h_1)^2}
$$
其中：

- $N$: 来自排队论的最小卫星数量（$N_{min}$）。
- $\lambda_{eff}$: 有效任务到达率。
- $R_1, R_2, h_1, h_2$: 关于卫星高度的几何参数。
