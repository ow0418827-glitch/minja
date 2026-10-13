# SCAR(谱域协同自适应精修)数学公式与方法

整理日期：2026年10月9日。依据当前项目的实际实现、配置与已落盘实验记录整理；数学表达用于说明现有实现，没有修改优化器或重新运行实验。

SCAR 是本项目对组合方案的工作名称，不对应一篇独立发表的论文。频谱模拟基础 SSA(频谱模拟攻击)来自[《对抗攻击的频域模型增强》官方论文页面](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/974_ECCV_2022_paper.php)，论文收录于 ECCV(欧洲计算机视觉会议)2022 会议论文集，属于会议论文；会议在 [CCF(中国计算机学会)官方条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/bl/2017-04-25/592064.shtml)中列为 B 类。项目在频谱模拟上加入自己的图像目标、梯度混合、动量、调度、早停与幅度选择，不能将全部组合方法归于原论文。

## 1. 任务、符号和约束

甲商品是指定错误购买目标，乙商品是用户附图希望购买的商品。算法修改甲图的检索表示，使它在乙图查询下更容易召回甲商品历史。

| 符号 | 含义 | 当前主要设置 |
|---|---|---|
| \(x_A\) | 原始甲图，通道值归一化至 \([0,1]\) | 始终作为扰动预算参照 |
| \(x_B\) | 目标乙图 | 目标特征预先计算 |
| \(x_t\) | 完成 \(t\) 次更新后的浮点候选图 | 原图起步或已审核旧图起步 |
| \(d\) | 像素通道总数 | \(3HW\) |
| \(T\) | 请求的最大更新次数 | 快速入口默认 300 |
| \(N\) | 每步频谱随机样本数 | 快速入口默认 5 |
| \(\varepsilon\) | 最大逐通道扰动 | \(16/255\) |
| \(w_{t,i}\) | 第 \(i\) 个代理的全局项权重 | 三个代理，动态更新 |
| \(\lambda_{\ell}\) | 局部相似度权重 | 0.2 |
| \(\beta\) | 未模拟当前候选梯度比例 | 0.5 |
| \(\lambda_R\) | 均方修改惩罚权重 | 0.01 |

允许的图片集合为

\[
\mathcal C=\{x:0\le x\le1,\ \|x-x_A\|_\infty\le\varepsilon\}.
\]

每个通道最多改变 16 个八位灰度级。即使从旧优化图开始精修，预算仍相对原始甲图计算。

源代码：[预算和初始化](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:278>)、[基础配置](<D:/Material/Literature Reading/code/dual-rag/configs/optimization.json:7>)。

## 2. 三代理全局余弦对齐

使用三个 CLIP(对比语言图像预训练模型)代理。设第 \(i\) 个代理的全局图像特征为 \(f_i(x)\)，全局相似度为

\[
G_i(x)=\frac{f_i(x)^\top f_i(x_B)}{\|f_i(x)\|_2\|f_i(x_B)\|_2}.
\]

当前代理分别是基础十六像素图块模型，以及采用两套预训练权重的基础三十二像素图块模型。三套模型均参与图像优化；最终检索分支不需要同时使用三套代理。

图像优化采用相似度最大化：

\[
J_t(x)=\sum_{i=1}^{3}w_{t,i}G_i(x)+\frac{\lambda_{\ell}}3\sum_{i=1}^{3}L_i(x).
\]

\(L_i\) 是下文的局部最优传输相似度。动态权重只用于全局项，局部项始终是三代理等权平均。代码虽然使用“损失”变量名，更新符号是加号，执行梯度上升。

图像优化的文本目标权重强制为零；后续图文检索中的文字参与不意味着这里有文本对齐损失。

源代码：[目标计算](<D:/Material/Literature Reading/code/dual-rag/src/dual_rag/optimization/attack_core/objective.py:194>)、[强制关闭文本项](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:215>)。

## 3. 局部聚类与最优传输

每个代理抽取局部图块特征 \(p_n\)，使用余弦相似度分配到 \(K=10\) 个中心：

\[
a_n^{(r)}=\arg\max_k\frac{p_n^\top c_k^{(r)}}{\|p_n\|_2\|c_k^{(r)}\|_2},
\qquad
c_k^{(r+1)}=\frac{\sum_n\mathbf1[a_n^{(r)}=k]p_n}{\sum_n\mathbf1[a_n^{(r)}=k]}.
\]

采用确定性的等间距图块索引初始化，运行 100 次聚类迭代，空簇保留旧中心。最终中心单位化为 \(\widehat c_k\)。分配索引不参与求导，中心的成员平均仍可传递梯度。

甲候选与乙图的中心余弦矩阵和传输代价为

\[
S_{ab}=\widehat c_a(x)^\top\widehat c_b(x_B),\qquad C_{ab}=1-S_{ab}.
\]

以均匀边缘质量 \(\mathbf a,\mathbf b\)近似求解熵正则最优传输：

\[
\min_{P\ge0}\ \sum_{a,b}P_{ab}C_{ab}
+\tau\sum_{a,b}P_{ab}(\log P_{ab}-1),
\quad P\mathbf1=\mathbf a,\quad P^\top\mathbf1=\mathbf b.
\]

Sinkhorn(辛克霍恩迭代)通过传输核和交替缩放近似求解：

\[
K_{ab}^{\mathrm{核}}=\max\{\exp[-C_{ab}/\max(\tau,10^{-6})],10^{-12}\},
\]

\[
u^{(r+1)}=\frac{\mathbf a}{K^{\mathrm{核}}v^{(r)}},\qquad
v^{(r+1)}=\frac{\mathbf b}{(K^{\mathrm{核}})^\top u^{(r+1)}},\qquad
P=\operatorname{diag}(u)K^{\mathrm{核}}\operatorname{diag}(v).
\]

除法均逐分量执行，分母下限为 \(10^{-12}\)。当前 \(\tau=0.1\)，最多 100 次缩放；若 \(\operatorname{mean}|u^{(r+1)}-u^{(r)}|<0.01\)则提前结束。该条件检查左缩放向量变化，不能表述为边缘质量残差已经收敛。

局部相似度为

\[
L_i(x)=\sum_{a,b}P_{ab}S_{ab}.
\]

传输计划 \(P\) 默认切断梯度。每次前向重新计算匹配，但反向传播将当前匹配计划视为常量；没有对完整传输求解过程求导。

源代码：[聚类](<D:/Material/Literature Reading/code/dual-rag/src/dual_rag/optimization/attack_core/ot.py:7>)、[传输计划](<D:/Material/Literature Reading/code/dual-rag/src/dual_rag/optimization/attack_core/ot.py:39>)、[局部相似度](<D:/Material/Literature Reading/code/dual-rag/src/dual_rag/optimization/attack_core/ot.py:72>)。

## 4. 动态代理权重

定义八位量化算子

\[
Q(x)=\frac{\operatorname{round}(255x)}{255},
\]

其中取整表示取最近整数。代理权重初始化与更新为

\[
w_{0,i}=\frac13,\qquad
w_{t+1,i}=\frac{\exp[-G_i(Q(x_t))]}{\sum_{j=1}^{3}\exp[-G_j(Q(x_t))]}.
\]

相似度较低的代理得到更高全局权重。时间顺序是：用 \(w_t\) 更新到 \(x_{t+1}\)，再根据更新前的量化图 \(Q(x_t)\)设置下一轮权重。权重相对最新候选有一步延迟。

源代码：[权重初始化](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:275>)、[权重更新](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:398>)。

## 5. 频谱随机模拟

采用非归一化二型 DCT(离散余弦变换)，单轴定义为

\[
[\mathcal D_L(z)]_k=2\sum_{n=0}^{L-1}z_n\cos\left[\frac\pi L\left(n+\frac12\right)k\right].
\]

二维变换沿高、宽分别执行一次，直接二维求和形式相应带系数 4。\(\mathcal D^{-1}\)为匹配的逆变换。

每次随机模拟首先加独立零均值高斯噪声，然后对全部频谱系数施加独立连续均匀缩放：

\[
\eta_{t,n}\sim\mathcal N\left(0,\left(\frac{\sigma_t}{255}\right)^2I\right),\qquad
M_{t,n,j}\sim\mathcal U(1-\rho_t,1+\rho_t),
\]

\[
\widetilde x_{t,n}=\mathcal D^{-1}\left[\mathcal D(x_t+\eta_{t,n})\odot M_{t,n}\right].
\]

\(\odot\)表示逐系数乘法。开始时 \(\rho=0.5\)，所以频谱乘数介于 0.5 和 1.5；没有二值频带丢弃，也没有把优化变量限制在低频系数子空间。

模拟样本不预先裁剪到合法像素区间，完整候选更新后才投影。优化变量始终为原尺寸全部像素，最终图片不保证严格限带。

源代码：[变换](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:140>)、[随机模拟](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:176>)。

## 6. 多次模拟平均与未模拟梯度协同

先将各模拟结果视为独立求梯度输入：

\[
g_t^{\mathrm{模拟}}=\frac1N\sum_{n=1}^{N}\left.\nabla_zJ_t(z)\right|_{z=\widetilde x_{t,n}},
\qquad g_t^{\mathrm{当前}}=\nabla_xJ_t(x_t).
\]

\(g_t^{\mathrm{当前}}\)是当前候选未经随机变换时的梯度，不是原始甲图 \(x_A\)的梯度。

模拟变换与原候选的计算图被切断，所以 \(g_t^{\mathrm{模拟}}\)没有乘随机变换的雅可比矩阵。不能直接将它写成对原变量严格求导的随机变换期望损失梯度。

加入相对原甲图、按预算归一化的均方修改惩罚：

\[
R(x)=\frac{\|x-x_A\|_2^2}{d\varepsilon^2},
\]

\[
g_t=(1-\beta)g_t^{\mathrm{模拟}}+\beta g_t^{\mathrm{当前}}
-\frac{2\lambda_R}{d\varepsilon^2}(x_t-x_A).
\]

快速方案 \(N=5\)、\(\beta=0.5\)、\(\lambda_R=0.01\)。每步求五份模拟梯度和一份未模拟梯度；每份梯度均按三个代理顺序求取，累计后释放各模型计算图。这是节省显存的执行方法，不改变三代理加和公式。

惩罚用于更新方向，不加入下文的固定候选评分。

源代码：[采样求梯度](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:366>)、[协同与惩罚](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:378>)、[逐代理求梯度](<D:/Material/Literature Reading/code/dual-rag/src/dual_rag/optimization/attack_core/objective.py:233>)。

## 7. 梯度归一化、动量、符号更新与投影

梯度按全部通道的平均绝对值归一化：

\[
\widehat g_t=\frac{g_t}{\max\{d^{-1}\sum_{j=1}^d|g_{t,j}|,\zeta\}},
\]

\(\zeta\)为所用浮点类型的最小正规正数。该分母不是欧氏范数或最大范数。

动量与更新为

\[
m_{t+1}=\mu_tm_t+\widehat g_t,\quad m_0=0,
\qquad
x_{t+1}=\Pi_{\mathcal C}\left[x_t+\alpha_t\operatorname{sign}(m_{t+1})\right].
\]

符号函数逐通道返回正、负或零。投影就是逐通道截断：

\[
l_j=\max(0,x_{A,j}-\varepsilon),\quad u_j=\min(1,x_{A,j}+\varepsilon),
\qquad [\Pi_{\mathcal C}(z)]_j=\max(l_j,\min(z_j,u_j)).
\]

这里的动量是本项目扩展；被参考的上游默认攻击脚本没有启用其注释中的动量代码。

源代码：[更新与投影](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:392>)。

## 8. 线性退火

一般调度用请求最大步数 \(T\)和完成次数 \(t\)：

\[
p_t=\frac{t}{\max(1,T-1)},\qquad
r_t=\begin{cases}0,&s=1,\\ \min(1,\max(0,(p_t-s)/(1-s))),&s<1.\end{cases}
\]

\(s\)表示退火开始比例；对步长倍率、动量、频谱掩码幅度和噪声标准差，均使用

\[
q_t=q_{\mathrm{起}}+(q_{\mathrm{终}}-q_{\mathrm{起}})r_t.
\]

快速配置采用 \(s=0\)，因此从第一步开始贯穿全程线性退火，而不是到后期才开启：

| 参数 | 起点 | 终点 |
|---|---:|---:|
| 更新步长 \(\alpha_t\) | \(0.5/255\) | \(0.1/255\) |
| 动量系数 \(\mu_t\) | 1.0 | 0.8 |
| 频谱幅度 \(\rho_t\) | 0.5 | 0.1 |
| 模拟噪声标准差 \(\sigma_t\)，八位灰度级 | 16 | 4 |

调度依据请求预算，提前终止不会把已经执行的调度重新压缩为全程。

源代码：[调度](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:127>)、[快速配置](<D:/Material/Literature Reading/code/minja/src/configs/ssa_fast.json:1>)、[快速入口](<D:/Material/Literature Reading/code/minja/run_optimized_ssa_126.ps1:1>)。

## 9. 固定量化评分与容忍区间内选优

当前量化候选用固定三代理等权评分：

\[
S_t=\frac13\sum_{i=1}^3G_i(Q(x_t))+\frac{0.2}3\sum_{i=1}^3L_i(Q(x_t)),
\qquad B_t=\max_{0\le u\le t}S_u.
\]

该分数不使用动态权重，不包含修改惩罚，不使用随机模拟图。这保证各步分数可比较。

实际完成更新数为 \(T_c\)，容忍度 \(\delta=0.002\)。从得分距离最高值不超过 \(\delta\)的量化候选中，保存修改最小者：

\[
t^*\in\arg\min_{\{u:S_u\ge B_{T_c}-\delta\}}
\frac1d\|Q(x_u)-x_A\|_2^2.
\]

并列时优先较高评分，再优先较早步数。实现保留评分与修改量之间未被支配的候选，因此不必在显存里存放全部历史图。

“历史最高评分步”“实际保存步”和“总完成步”是三个不同字段。最高评分 \(B_t\)独立保留，供早停使用。

源代码：[固定评分](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:303>)、[候选保留](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:313>)、[最终选优](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:414>)。

## 10. 平台早停

使用长度 \(W=50\)的非重叠检查窗口：

\[
\Delta_t=B_t-B_{t-W}.
\]

完成至少 200 步，且步数为 50 的倍数时才检查。连续低收益计数为

\[
c_t=\begin{cases}c_{\mathrm{上次检查}}+1,&\Delta_t<0.002,\\0,&\Delta_t\ge0.002.\end{cases}
\qquad c_t\ge2\ \Longrightarrow\ \text{停止更新}.
\]

默认最早在完成 250 步后停止：200 步与 250 步的两个连续窗口都低于门槛。任意微小新高仍更新 \(B_t\)，但只有窗口改善达到门槛才清空低收益计数。检查发生在下一轮频谱采样之前。

早停只表示当前代理收益较低，不能推出真实检索成功或购物成功。洗发水核心实际完成了全部 300 步，没有触发平台早停。

源代码：[早停器](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:95>)、[循环内检查](<D:/Material/Literature Reading/code/minja/src/vision/ssa_attack.py:347>)。

## 11. 全文图文检索

记单位化操作为 \(U(v)=v/\|v\|_2\)。完整历史按模型上下文长度分成不重叠文本块 \(C_r\)，每块有效词元数 \(n_r\)：

\[
h_T=U\left[\frac{\sum_r\max(1,n_r)U(E_T(C_r))}{\sum_r\max(1,n_r)}\right],
\qquad h_I=U(E_I(x)).
\]

CLIP 分支的记忆键与查询分别融合：

\[
k_j=U(a_kh_{I,j}+(1-a_k)h_{T,j}),\qquad
q_r=U(a_qh_{I,B}+(1-a_q)h_{T,r}^{Q}),
\]

\[
s_{rj}=q_r^\top k_j.
\]

\(a_k,a_q\)可独立设置。0.5 时实际保留 \(U(h_I+h_T)\) 的浮点运算路径。融合系数是归一化前的向量系数，不能直接解释为最终分数的模态百分比。

若只看首位检索，指定甲历史 \(a\)的余量为

\[
m_r=s_{ra}-\max_{j\ne a}s_{rj},\qquad m_{\min}=\min_r m_r.
\]

\(m_r>0\)说明甲历史严格领先所有其他候选；一般排序按分数降序，相同分数按原记忆索引排序。实际实验库含 14 条正常历史和 1 条指定甲历史，固定查询集为同一乙图配 20 条文字改写。

源代码：[全文分块](<D:/Material/Literature Reading/code/minja/src/vision/joint_encoder.py:118>)、[图文融合](<D:/Material/Literature Reading/code/minja/src/vision/joint_encoder.py:152>)、[正式排序](<D:/Material/Literature Reading/code/minja/src/vision/visual_retrieval.py:246>)、[离线余量](<D:/Material/Literature Reading/code/minja/src/experiments/probe_vision_optimization.py:51>)。

最新 GME(通用多模态嵌入模型)分支先对文本块与同一张图片做模型原生联合编码，再加权聚合：

\[
v_{IT}=U\left[\frac{\sum_r\max(1,n_r)U(E_{IT}(C_r,x))}{\sum_r\max(1,n_r)}\right].
\]

该分支不能套用 CLIP 的独立图像向量和文本向量相加公式。其两端参数 0.5 仅为接口兼容值，不代表手动各占一半。最新洗发水跨模型实验复用已有 SCAR 图，新增图像优化为零步。

源代码：[原生融合及全文聚合](<D:/Material/Literature Reading/code/minja/src/vision/joint_encoder_worker.py:67>)、[权重限制](<D:/Material/Literature Reading/code/minja/src/vision/joint_encoder.py:263>)、[跨模型实验说明](<D:/Material/Literature Reading/code/minja/src/output/scar_shampoo_gme_workflow_20261009/EXPERIMENT.md:14>)。

## 12. 检索余量内的扰动幅度选择

令 \(A\)为八位原甲图，\(X\)为核心优化后八位图，比例 \(r\)产生候选

\[
X_r=\operatorname{clip}_{[0,255]}\left(\operatorname{round}[A+r(X-A)]\right).
\]

缩小的是相对原图的扰动幅度，图片尺寸保持原样。三组扫描

\[
\mathcal R=\{1,0.875,0.75,0.625,0.5,0.375,0.25\};
\]

洗发水额外扫描 \(r=0\)。候选可行条件为

\[
\forall r_q\in\{1,\ldots,20\},\ \operatorname{rank}_{r_q}(a;X_r)=1,
\quad m_{\min}(X_r)\ge0.005,
\quad G_{\mathrm{检索模型}}(X_r)\ge c_{\mathrm{参考}}.
\]

三组参考来自同图对原 500 步攻击图片；洗发水参考来自同图对原 500 步频域图片。在可行比例集合 \(\mathcal F\)中选择

\[
r^*\in\arg\min_{r\in\mathcal F}\frac1d\sum_j(X_{r,j}-A_j)^2.
\]

每个八位候选均重新编码、重新统计误差、重新排名，不继承缩放前评分。量化后均方修改不严格遵循连续空间的 \(r^2\)关系。这里只是离散候选选择，没有额外梯度优化，也没有证明全局最小或未知查询可行。

三组无可行候选则记录空选择；洗发水无可行候选则回退 \(r=1\)并记录回退标识，回退不能算满足门槛。

| 已有流程 | 幅度选择两端图像系数 | 后续购物两端图像系数 | 实际选中比例 |
|---|---:|---:|---|
| 相机、椅子、牛仔裤三组 | 0.9 | 0.5 | 0.875、0.75、0.625 |
| 洗发水 CLIP 流程 | 0.5 | 0.5 | 0.625 |
| 最新洗发水 GME 流程 | 复用已选图片 | 原生联合编码 | 复用 0.625 图片 |

选择阶段可行不保证改变检索权重或竞争记忆库后仍可行。三组相机和椅子核心图使用旧图起点额外精修 30 步；牛仔裤和洗发水快速核心从原甲图开始，最多 300 步。因此“300 步”不是所有历史 SCAR 输出的共同生成预算。

源代码：[三组可行与选优](<D:/Material/Literature Reading/code/minja/src/experiments/select_perturbation_scale.py:168>)、[洗发水可行与选优](<D:/Material/Literature Reading/code/minja/src/experiments/scar_shampoo_experiment.py:106>)、[三组复用配置](<D:/Material/Literature Reading/code/minja/src/experiments/reused_scar_pipeline.py:79>)。

## 13. 像素审计和实验指标

所有像素误差以八位通道差 \(\Delta_j=X_{r,j}-A_j\)计算：

\[
E_1=\frac1d\sum_j|\Delta_j|,\quad E_2=\frac1d\sum_j\Delta_j^2,\quad
D_\infty=\max_j|\Delta_j|\le16.
\]

\(E_1\)是平均绝对误差，\(E_2\)是均方误差。三组工具还记录峰值信噪比与达到预算上限的通道比例：

\[
P_{\mathrm{信噪比}}=10\log_{10}\frac{255^2}{E_2},\qquad
f_{\mathrm{上限}}=\frac1d\sum_j\mathbf1[|\Delta_j|=16].
\]

\(E_2=0\)时报告将信噪比记为空值。源图、保存后图片都要重新解码核验，且模型权重、预处理、精度、解码器与缓存身份需要匹配；这些是复现与预算审核，不是额外目标项。

检索首位率和实际误购率分别为

\[
R_{\mathrm{首位}}=\frac1n\sum_{r=1}^n\mathbf1[\operatorname{rank}_{r}(a)=1],
\qquad
R_{\mathrm{误购}}=\frac1n\sum_{r=1}^n\mathbf1[\text{完成购买且最终商品为甲}].
\]

乙图正确购买率同理统计最终商品为乙的完成回合。首次检索首位、真正呈现甲历史、完成误购甲是不同事件；它们不能相互替代。正式购物率在全部计划回合完成且无技术错误时才给出，否则为空；部分阶段指标另按已测量回合计算，不能混用分母。

源代码：[像素审计](<D:/Material/Literature Reading/code/minja/src/experiments/select_perturbation_scale.py:43>)、[购物指标](<D:/Material/Literature Reading/code/minja/src/experiments/nine_group_experiment.py:1291>)、[完整性与分母](<D:/Material/Literature Reading/code/minja/src/experiments/nine_group_experiment.py:1332>)。

## 14. 实现流程

1. 绑定真实甲商品历史、原甲图及独立乙图，冻结记忆库与查询集合。
2. 加载三个代理，预计算乙图全局特征、局部特征与聚类中心。
3. 当前候选量化后按固定分数评价，更新历史最高和容忍区间候选。
4. 检查平台早停；未停止则计算当前步的退火参数。
5. 求五份频谱模拟梯度、一份未模拟梯度，混合并减去修改惩罚梯度。
6. 按平均绝对值归一化，累计动量，做符号更新和预算投影。
7. 根据更新前量化图的全局相似度设置下一步代理权重。
8. 完成预算或早停后，保存容忍区间内修改量最小候选，并重新解码审计。
9. 扫描离散扰动幅度，在固定库、固定查询和明确检索设置下选择可行最小修改图。
10. 导入选图、重建检索向量，独立评估检索和购物结果。图片复用导入的零步表示没有新增优化，核心生成成本仍需按来源记录计算。

## 15. 三种分数的区别

| 数量 | 使用位置 | 组成 | 是否受动态权重影响 |
|---|---|---|---|
| \(J_t\) | 求各输入图像的梯度 | 动态加权全局项 + 等权局部项 | 是，全局项 |
| \(S_t\) | 量化候选选择、平台早停 | 三代理等权全局项 + 等权局部项 | 否 |
| \(s_{rj}\) | 实际记忆检索、余量与幅度选择 | 查询向量与记忆键的点积 | 使用独立检索配置 |

均方修改惩罚另用于生成更新方向；最终购物结果由真实执行轨迹定义，不能由上述任一种分数直接推断。
