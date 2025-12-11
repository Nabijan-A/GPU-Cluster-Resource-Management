# 第3章 GPU分配策略与定量建模

## 3.1 GPU内存原理与容量管型

### 3.1.1 GPU内存架构

GPU内存（显存，VRAM）是深度学习训练的关键资源。现代GPU采用分层内存架构：

```
┌─────────────────────────────────────┐
│   寄存器文件 (Registers)             │  最快，容量极小
│   ~256KB per SM                     │
├─────────────────────────────────────┤
│   共享内存 (Shared Memory)           │  快速，可编程
│   ~48-164KB per SM                  │
├─────��───────────────────────────────┤
│   L1缓存 (L1 Cache)                 │  自动管理
│   ~128KB per SM                     │
├─────────────────────────────────────┤
│   L2缓存 (L2 Cache)                 │  所有SM共享
│   ~40-60MB                          │
├─────────────────────────────────────┤
│   全局内存 (Global Memory/VRAM)      │  主要存储空间
│   8GB - 80GB+                       │
└─────────────────────────────────────┘
```

**关键特性：**
- **带宽**：现代GPU显存带宽可达 900-3000 GB/s
- **延迟**：全局内存访问延迟 ~200-800 个时钟周期
- **容量**：训练大模型的主要瓶颈

### 3.1.2 GPU内存占用组成

深度学习训练中的GPU内存占用可分解为：

$$
M_{total} = M_{model} + M_{optimizer} + M_{activation} + M_{gradient} + M_{workspace}
$$

**详细分解：**

1. **模型参数内存** $M_{model}$：
   $$
   M_{model} = N_{params} \times S_{dtype}
   $$
   - $N_{params}$：模型参数总数
   - $S_{dtype}$：数据类型字节数（FP32=4, FP16=2, INT8=1）

2. **优化器状态内存** $M_{optimizer}$：
   
   对于Adam优化器：
   $$
   M_{optimizer} = 2 \times M_{model} + M_{model} = 3 \times M_{model}
   $$
   - 动量项（momentum）：$M_{model}$
   - 二阶矩估计（variance）：$M_{model}$
   - 梯度副本：$M_{model}$

3. **激活值内存** $M_{activation}$：
   $$
   M_{activation} = \sum_{l=1}^{L} (B \times H_l \times W_l \times C_l \times S_{dtype})
   $$
   - $B$：批次大小（batch size）
   - $H_l, W_l, C_l$：第$l$层的高度、宽度、通道数
   - $L$：网络层数

4. **梯度内存** $M_{gradient}$：
   $$
   M_{gradient} = M_{model}
   $$

5. **工作空间内存** $M_{workspace}$：
   - CUDA内核临时缓冲区
   - 通信缓冲区（分布式训练）
   - 通常为模型大小的10-20%

### 3.1.3 混合精度训练内存优化

使用FP16/BF16混合精度训练：

$$
M_{mixed} = M_{model}^{FP16} + M_{model}^{FP32} + M_{optimizer}^{FP32} + M_{activation}^{FP16}
$$

**内存节省比例：**
$$
R_{save} = 1 - \frac{M_{mixed}}{M_{full}} \approx 0.4 \sim 0.5
$$

即可节省40-50%的内存。

## 3.2 批次大小与内存关系建模

### 3.2.1 批次大小对内存的影响

批次大小$B$与GPU内存占用呈近似线性关系（当$B$较大时）：

$$
M(B) = M_{static} + k \cdot B
$$

其中：
- $M_{static}$：静态内存（模型、优化器状态、梯度）
- $k$：每个样本的内存系数（主要是激活值）

**激活值内存与批次大小的关系：**

对于Transformer模型：
$$
M_{activation}(B) = B \times L \times \left( 4 \times d_{model} \times seq\_len + 2 \times seq\_len^2 \right) \times S_{dtype}
$$

- $L$：Transformer层数
- $d_{model}$：隐藏维度
- $seq\_len$：序列长度

### 3.2.2 最大批次大小估算

给定GPU内存容量$M_{GPU}$，最大批次大小：

$$
B_{max} = \left\lfloor \frac{M_{GPU} - M_{static} - M_{reserved}}{k} \right\rfloor
$$

- $M_{reserved}$：预留内存（通常10-15%）

**实际计算示例（GPT-2模型）：**

假设：
- GPU内存：16GB
- 模型参数：1.5B（FP16）= 3GB
- 优化器状态（Adam）：9GB
- 预留内存：1.6GB
- 每样本激活值：~50MB

$$
B_{max} = \left\lfloor \frac{16000 - 3000 - 9000 - 1600}{50} \right\rfloor = \left\lfloor \frac{2400}{50} \right\rfloor = 48
$$

### 3.2.3 梯度累积策略

当目标批次大小$B_{target} > B_{max}$时，使用梯度累积：

$$
N_{accum} = \left\lceil \frac{B_{target}}{B_{max}} \right\rceil
$$

**有效批次大小：**
$$
B_{effective} = B_{micro} \times N_{accum}
$$

**训练时间影响：**
$$
T_{epoch} = \frac{N_{samples}}{B_{effective}} \times t_{iter} \times N_{accum}
$$

其中$t_{iter}$是单次迭代时间。

## 3.3 多维度资源需求建模

### 3.3.1 四维资源模型

GPU集群资源可建模为四维向量：

$$
\mathbf{R} = (R_{GPU}, R_{MEM}, R_{CPU}, R_{BW})
$$

- $R_{GPU}$：GPU计算能力（TFLOPS）
- $R_{MEM}$：GPU内存容量（GB）
- $R_{CPU}$：CPU核心数
- $R_{BW}$：网络带宽（Gbps）

### 3.3.2 任务资源需求函数

对于训练任务$j$，资源需求可表示为：

$$
\mathbf{D}_j = \left( D_j^{GPU}, D_j^{MEM}, D_j^{CPU}, D_j^{BW} \right)
$$

**GPU计算需求：**
$$
D_j^{GPU} = \frac{2 \times N_{params} \times B \times seq\_len \times N_{layers}}{t_{target}}
$$

**内存需求：**
$$
D_j^{MEM} = (1 + k_{opt}) \times N_{params} \times S_{dtype} + M_{activation}(B)
$$

- $k_{opt}$：优化器内存系数（Adam: 3, SGD: 1）

**带宽需求（多GPU训练）：**
$$
D_j^{BW} = \frac{2 \times N_{params} \times S_{dtype} \times N_{GPU}}{t_{comm}}
$$

### 3.3.3 资源利用率矩阵

定义集群的资源利用率矩阵：

$$
\mathbf{U} = \begin{bmatrix}
u_{1,GPU} & u_{1,MEM} & u_{1,CPU} & u_{1,BW} \\
u_{2,GPU} & u_{2,MEM} & u_{2,CPU} & u_{2,BW} \\
\vdots & \vdots & \vdots & \vdots \\
u_{n,GPU} & u_{n,MEM} & u_{n,CPU} & u_{n,BW}
\end{bmatrix}
$$

其中：
$$
u_{i,r} = \frac{\text{已使用资源}_r}{\text{总资源}_r} \in [0, 1]
$$

**资源碎片化度量：**
$$
F = \frac{1}{n} \sum_{i=1}^{n} \text{std}(u_{i,GPU}, u_{i,MEM}, u_{i,CPU}, u_{i,BW})
$$

$F$越大，资源碎片化越严重。

## 3.4 优化目标与约束条件

### 3.4.1 多目标优化框架

GPU资源分配可建模为多目标优化问题：

$$
\begin{align}
\min_{\mathbf{x}} \quad & \mathbf{F}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x})) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, 2, \ldots, p \\
& h_j(\mathbf{x}) = 0, \quad j = 1, 2, \ldots, q
\end{align}
$$

### 3.4.2 主要优化目标

**1. 最小化平均等待时间：**
$$
f_1(\mathbf{x}) = \frac{1}{|J|} \sum_{j \in J} (t_j^{start} - t_j^{submit})
$$

**2. 最大化吞吐量：**
$$
f_2(\mathbf{x}) = -\frac{|J_{completed}|}{T_{total}}
$$

**3. 最大化资源利用率：**
$$
f_3(\mathbf{x}) = -\frac{1}{T} \int_0^T \frac{\sum_{i=1}^{n} R_i^{used}(t)}{\sum_{i=1}^{n} R_i^{total}} dt
$$

**4. 最小化能耗：**
$$
f_4(\mathbf{x}) = \sum_{j \in J} \sum_{g \in G_j} P_g \times t_j^{exec}
$$

- $P_g$：GPU功耗
- $G_j$：任务$j$使用的GPU集合

**5. 保证公平性（Jain's Fairness Index）：**
$$
f_5(\mathbf{x}) = -\frac{\left(\sum_{u \in U} \theta_u\right)^2}{|U| \times \sum_{u \in U} \theta_u^2}
$$

- $\theta_u$：用户$u$获得的资源份额

### 3.4.3 约束条件

**1. 资源容量约束：**
$$
\sum_{j \in J_i(t)} d_{j,r} \leq C_{i,r}, \quad \forall i \in N, r \in R, t \in T
$$

**2. 任务完整性约束：**
$$
\sum_{i \in N} x_{ij} \times n_{ij} = n_j^{required}, \quad \forall j \in J
$$

- $x_{ij} \in \{0, 1\}$：任务$j$是否分配到节点$i$
- $n_{ij}$：分配的GPU数量

**3. 亲和性约束（数据局部性）：**
$$
\sum_{i \in S_j} x_{ij} \geq \alpha \times \sum_{i \in N} x_{ij}
$$

- $S_j$：存储任务$j$数据的节点集
- $\alpha$：局部性系数（如0.8）

**4. GPU拓扑约束：**

对于需要多GPU的任务，优先同节点分配：
$$
\sum_{i \in N} \mathbb{1}(x_{ij} = 1) \leq \beta
$$

- $\beta$：允许的最大节点数（通常为1或2）

**5. SLA约束：**
$$
t_j^{complete} - t_j^{submit} \leq SLA_j, \quad \forall j \in J_{critical}
$$

### 3.4.4 加权目标函数

将多目标转化为单目标：

$$
F(\mathbf{x}) = \sum_{i=1}^{m} w_i \times \frac{f_i(\mathbf{x}) - f_i^{min}}{f_i^{max} - f_i^{min}}
$$

权重满足：$\sum_{i=1}^{m} w_i = 1, \quad w_i \geq 0$

**动态权重调整策略：**
$$
w_i(t) = w_i^{base} \times \left(1 + \gamma \times \frac{f_i(t) - \bar{f}_i}{\sigma_i}\right)
$$

- $\gamma$：调整系数
- $\bar{f}_i$：历史平均值
- $\sigma_i$：标准差

## 3.5 定量分析方法

### 3.5.1 排队论模型

将GPU集群建模为M/M/c排队系统：

**到达率：** $\lambda$ (任务/小时)  
**服务率：** $\mu$ (任务/小时/GPU)  
**GPU数量：** $c$

**系统利用率：**
$$
\rho = \frac{\lambda}{c \times \mu}
$$

**平均等待时间（Erlang-C公式）：**
$$
W_q = \frac{C(c, \lambda/\mu)}{c \times \mu - \lambda} \times \frac{1}{\mu}
$$

其中：
$$
C(c, a) = \frac{\frac{a^c}{c!} \times \frac{c}{c-a}}{\sum_{k=0}^{c-1} \frac{a^k}{k!} + \frac{a^c}{c!} \times \frac{c}{c-a}}
$$

**平均队列长度：**
$$
L_q = \lambda \times W_q
$$

### 3.5.2 效率评估指标

**1. GPU利用率：**
$$
U_{GPU} = \frac{\sum_{i=1}^{n} \sum_{t=0}^{T} \text{active}_{i}(t)}{n \times T} \times 100\%
$$

**2. 作业吞吐量：**
$$
\Theta = \frac{N_{completed}}{T_{window}}
$$

**3. 资源效率（FLOPS利用率）：**
$$
E_{FLOPS} = \frac{\sum_{j} FLOPS_j^{actual}}{\sum_{j} FLOPS_j^{peak}} \times 100\%
$$

**4. 扩展效率（多GPU训练）：**
$$
E_{scale}(n) = \frac{T_1}{n \times T_n} \times 100\%
$$

- $T_1$：单GPU训练时间
- $T_n$：$n$个GPU训练时间

**理想情况：** $E_{scale}(n) = 100\%$（线性扩展）

**5. 内存带宽利用率：**
$$
U_{BW} = \frac{\text{实际数据传输量}}{\text{理论峰值带宽} \times \text{时间}} \times 100\%
$$

### 3.5.3 性能建模与预测

**Amdahl定律（并行加速比）：**
$$
S(n) = \frac{1}{(1-p) + \frac{p}{n}}
$$

- $p$：可并行部分比例
- $n$：处理器数量

**通信开销模型（α-β模型）：**
$$
T_{comm} = \alpha + \beta \times M
$$

- $\alpha$：延迟（latency）
- $\beta$：每字节传输时间
- $M$：消息大小

**数据并行训练时间模型：**
$$
T_{epoch}(n) = \frac{T_{compute}}{n} + T_{comm}(n) + T_{sync}
$$

其中：
$$
T_{comm}(n) = \log_2(n) \times \left(\alpha + \beta \times \frac{2(n-1)}{n} \times M_{model}\right)
$$

**模型并行训练时间：**
$$
T_{model}(n) = \frac{T_{compute}}{n} + (L-1) \times T_{bubble}
$$

- $L$：流水线阶段数
- $T_{bubble}$：流水线气泡时间

### 3.5.4 碎片化分析

**内存碎片率：**
$$
F_{mem} = 1 - \frac{\max(\text{可分配连续内存})}{\text{总空闲内存}}
$$

**GPU碎片化指数：**
$$
I_{frag} = \frac{\sum_{i=1}^{n} (\text{空闲GPU}_i \times \text{不可用标志}_i)}{\sum_{i=1}^{n} \text{空闲GPU}_i}
$$

**时间碎片（任务间隙）：**
$$
F_{time} = \frac{\sum_{k=1}^{K} (t_k^{start} - t_{k-1}^{end})}{T_{total}}
$$

## 3.6 分配算法复杂度分析

### 3.6.1 问题复杂性

GPU资源分配问题是NP-hard问题（可规约到多维装箱问题）。

**证明思路：**
- 多维装箱问题：给定$n$个物品和$m$个箱子，每个物品有$d$维资源需求
- GPU分配：任务（物品）需要分配到节点（箱子），满足GPU、内存、CPU等多维约束
- 决策版本：是否存在可行分配方案？→ NP-complete

### 3.6.2 算法时间复杂度

**1. 暴力搜索：**
$$
O(m^n)
$$
- $n$：任务数
- $m$：节点数

**2. 贪心算法（First-Fit, Best-Fit）：**
$$
O(n \times m)
$$

**3. 遗传算法：**
$$
O(G \times P \times n \times m)
$$
- $G$：代数
- $P$：种群大小

**4. 动态规划（特定场景）：**
$$
O(n \times C)
$$
- $C$：资源容量离散化级别

### 3.6.3 近似比分析

**First-Fit Decreasing (FFD) 算法：**

对于一维装箱问题：
$$
FFD(I) \leq \frac{11}{9} \times OPT(I) + \frac{6}{9}
$$

多维情况下，近似比退化到：
$$
FFD_d(I) \leq (2 - \epsilon) \times OPT(I)
$$

## 3.7 分配策略框架图

```
┌───────────────────────────────────────────────────────────┐
│              GPU资源分配决策系统                            │
└───────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  任务分析层   │   │  资源监控层   │   │  策略配置层   │
│              │   │              │   │              │
│ • 资源需求   │   │ • GPU状态    │   │ • 优化目标   │
│ • 优先级     │   │ • 内存使用   │   │ • 约束条件   │
│ • SLA要求    │   │ • 网络拓扑   │   │ • 权重参数   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
              ┌───────────────────────┐
              │   调度算法引擎         │
              │                       │
              │  ┌─────────────────┐  │
              │  │ 预处理模块      │  │
              │  │ • 任务排序      │  │
              │  │ • 资源过滤      │  │
              │  └────────┬────────┘  │
              │           ▼            │
              │  ┌─────────────────┐  │
              │  │ 匹配算法        │  │
              │  │ • 贪心          │  │
              │  │ • 遗传算法      │  │
              │  │ • 强化学习      │  │
              │  └────────┬────────┘  │
              │           ▼            │
              │  ┌─────────────────┐  │
              │  │ 优化模块        │  │
              │  │ • 碎片整理      │  │
              │  │ • 负载均衡      │  │
              │  └────────┬────────┘  │
              └───────────┼───────────┘
                          ▼
              ┌───────────────────────┐
              │   执行与反馈           │
              │                       │
              │  • 资源分配           │
              │  • 性能监控           │
              │  • 自适应调整         │
              └───────────────────────┘
```

## 3.8 案例研究：大模型训练的资源分配

### 3.8.1 问题设定

训练GPT-3级别模型（175B参数）：

**模型规格：**
- 参数量：$N = 175 \times 10^9$
- 隐藏维度：$d = 12288$
- 层数：$L = 96$
- 注意力头数：$h = 96$

**训练配置：**
- 批次大小：$B = 1536$
- 序列长度：$s = 2048$
- 混合精度（FP16）

### 3.8.2 内存需求计算

**模型参数内存：**
$$
M_{model} = 175 \times 10^9 \times 2 \text{ bytes} = 350 \text{ GB}
$$

**优化器状态（Adam）：**
$$
M_{opt} = 3 \times 350 = 1050 \text{ GB}
$$

**梯度：**
$$
M_{grad} = 350 \text{ GB}
$$

**激活值（检查点技术，每$k$层保存）：**
$$
M_{act} = B \times s \times d \times \frac{L}{k} \times 2 \approx \frac{1536 \times 2048 \times 12288 \times 96}{16} \times 2 \approx 360 \text{ GB}
$$

**总内存需求：**
$$
M_{total} = 350 + 1050 + 350 + 360 = 2110 \text{ GB}
$$

### 3.8.3 分配方案

使用A100 80GB GPU，采用3D并行：

**数据并行度：** $DP = 16$  
**张量并行度：** $TP = 8$  
**流水线并行度：** $PP = 8$

**总GPU数：**
$$
N_{GPU} = DP \times TP \times PP = 16 \times 8 \times 8 = 1024
$$

**每GPU内存占用：**
$$
M_{per\_gpu} = \frac{M_{total}}{TP \times PP} = \frac{2110}{64} \approx 33 \text{ GB}
$$

满足80GB容量要求。

**训练吞吐量估算：**
$$
\text{Throughput} = \frac{B \times s}{T_{iter}} \approx \frac{1536 \times 2048}{2.5} \approx 1.26 \times 10^6 \text{ tokens/s}
$$

### 3.8.4 成本分析

**训练时间（1 epoch，300B tokens）：**
$$
T_{epoch} = \frac{300 \times 10^9}{1.26 \times 10^6} \approx 238 \text{ hours} \approx 10 \text{ days}
$$

**GPU时成本（假设$1/GPU/hour）：**
$$
\text{Cost} = 1024 \times 10 \times 24 \times 1 = \$245,760
$$

**能耗（每GPU 400W）：**
$$
E = 1024 \times 0.4 \times 240 = 98,304 \text{ kWh}
$$

## 3.9 小结

本章建立了GPU资源分配的定量分析框架，包括：

1. **内存模型**：建立了GPU内存占用的详细数学模型，为批次大小选择提供理论依据
2. **资源建模**：构建多维度资源需求模型，综合考虑计算、内存、网络等资源
3. **优化框架**：定义了多目标优化问题，包含5类主要目标和5类约束条件
4. **定量方法**：引入排队论、性能建模、复杂度分析等数学工具
5. **实战案例**：通过GPT-3级别模型训练案例，展示了完整的资源规划流程

这些定量模型和方法将在后续章节中用于具体调度算法的设计和评估。

---

**关键公式速查：**

| 公式 | 说明 |
|------|------|
| $M_{total} = M_{model} + M_{opt} + M_{act} + M_{grad} + M_{ws}$ | 总内存占用 |
| $B_{max} = \lfloor \frac{M_{GPU} - M_{static}}{k} \rfloor$ | 最大批次大小 |
| $E_{scale}(n) = \frac{T_1}{n \times T_n}$ | 扩展效率 |
| $U_{GPU} = \frac{\sum \text{active}}{n \times T}$ | GPU利用率 |
| $T_{epoch}(n) = \frac{T_{comp}}{n} + T_{comm}(n)$ | 并行训练时间 |

**下一章预告：** 第4章将介绍主流调度算法，包括FIFO、Fair-Share、Backfilling等经典方法及其改进策略。
