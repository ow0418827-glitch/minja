# 可选频谱模拟攻击与旧乙图三组实验

项目新增 `ssa(频谱模拟攻击方法)`，保留 `frequency_momentum(频域动量方法)`、`frequency(原始低频方法)`、`legacy_pixel(原始像素方法)`和 `fair_pixel(公平像素基线方法)`。运行参数选择方法；已有冻结实验的输出目录不能直接切换方法，应为不同方法使用独立目录。

## 官方来源与实现范围

SSA(频谱模拟攻击)来自 [Frequency Domain Model Augmentation for Adversarial Attack(对抗攻击的频域模型增强)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640543.pdf)。该论文已收录于 ECCV(欧洲计算机视觉会议) 2022 会议论文集，属于会议论文，并非期刊论文；[大会官方接收清单](https://eccv2022.ecva.net/program/accepted-papers/)将论文编号 974 列在口头报告论文中。会议属于 CCF(中国计算机学会)推荐人工智能领域 B 类，见[学会官方会议条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/bl/2017-04-25/592064.shtml)。

实现参考[作者官方仓库](https://github.com/yuyang-long/SSA)，来源版本为 `c955cf07c8372bfc4e9f17e647042e027f9f3b1d(上游提交标识)`，主要参考 `attack.py(官方攻击脚本)`与 `dct.py(官方离散余弦变换实现)`。上游采用 Apache-2.0(阿帕奇第二版开源许可)，许可副本位于[第三方许可文件](third_party/SSA_LICENSE)。本地适配位于[频谱模拟优化模块](../src/vision/ssa_attack.py)。

官方核心操作及本地保留的行为如下：

1. 优化变量为完整像素图，不限制在低频系数子空间。
2. 每次模拟先在当前图像上添加标准差为 `sigma(高斯噪声标准差)`除以 255 的零均值高斯噪声，再做二维 DCT(离散余弦变换)，用区间 `[1-rho, 1+rho](连续频谱权重范围)`内的独立均匀随机权重乘频谱，最后做逆变换。`rho(频谱权重幅度)`默认 0.5，因此权重位于 0.5 至 1.5；官方机制不采用二值频带丢弃。
3. 模拟后的图像成为独立的求梯度输入，累计并平均多次模拟输入的梯度；按照官方脚本的处理，不通过随机变换的雅可比继续回传。模拟图本身不先裁剪到合法像素范围，约束作用于每次更新后的完整像素变量。

本项目保留原三个 CLIP(对比语言图像预训练模型)代理、目标图片预处理、全局余弦与局部最优传输组合目标、动态代理权重和原空间更新步长；文字目标权重仍为零。原论文使用分类损失，本地使用现有目标图表征对齐损失，因此属于面向项目的适配。官方默认脚本中的动量更新代码被注释，本项目增加可配置归一化梯度动量，默认 1.0，设为 0 可关闭累计。

每步更新仍投影到源图加减十六灰度级预算与合法像素范围的交集。候选先按八位图片表示评分，按固定等权全局余弦加原局部项保留最优候选；完成全部请求步数后保存图片，重新读取并核验整数像素预算。元数据分别记录完成步数和最优候选步数，以及有效参数、实现摘要、代理快照、官方来源和任务适配行为。

完整像素变量扩大了优化自由度，但不能据此保证目标表征余弦达到 0.95 以上、抢占 Top-1(检索首位)、普遍优于原方法或购物攻击成功。作者报告的 95.4% 是特定分类模型与数据集上的平均攻击成功率，不能作为本项目表征拟合上限；详见[论文与官方仓库](https://github.com/yuyang-long/SSA)。

## 运行参数

参数首次出现的中文说明如下；后面的命令保留可直接运行的原始参数拼写。

| 脚本参数／底层参数 | 默认值与范围 | 作用 |
|---|---|---|
| `-Method(方法参数)`／`--method(底层方法参数)` | 通用入口仍为原始像素方法；旧乙图专用入口仍为频域动量方法 | 显式传入 `ssa` 即启用频谱模拟。 |
| `-SsaSamples(频谱采样次数参数)`／`--ssa-samples(底层采样次数参数)` | 20，正整数 | 每个像素更新步骤的频谱模拟次数；每次仍计算三个代理梯度。 |
| `-SsaRho(频谱随机权重幅度参数)`／`--ssa-rho(底层随机权重幅度参数)` | 0.5，范围 [0, 1] | 随机频谱权重为 1 减该值到 1 加该值。 |
| `-SsaSigma(高斯噪声标准差参数)`／`--ssa-sigma(底层噪声标准差参数)` | 16，有限非负数，单位为八位灰度级 | 前向模拟噪声，与最终像素扰动预算是不同参数。 |
| `-SsaMomentum(像素梯度动量参数)`／`--ssa-momentum(底层像素梯度动量参数)` | 1.0，范围 [0, 1] | 归一化梯度的历史累计；0 关闭累计。 |

频谱模拟参数只接受用于 `ssa`，低频方法参数 `-FrequencyAxisRatio(轴向频带比例参数)`和 `-FrequencyMomentum(频率梯度动量参数)`只接受用于 `frequency_momentum`；两组参数互斥，错误搭配会报错。

其余入口参数包括 `-Stage(阶段参数)`、`-AttackSteps(图片优化步数参数)`、`-Root(独立输出目录参数)`、`-Port(本地商城端口参数)`、`-Device(运行设备参数)`与 `-SourceRoot(历史来源目录参数)`。旧乙图入口支持 `Prepare(准备阶段)`、`Run(完整运行阶段)`、`Check(检查阶段)`与 `Status(状态阶段)`，默认完整运行，图片优化步数为 500，端口为 3007，设备为 `cuda(图形处理器运行设备)`。

## 旧乙图第 1、2、6 组完整实验

本轮入口为[旧乙图三组脚本](../run_old_b_frequency_126.ps1)。它复用 `src/output/nine_groups_vague_top1_batch_126(旧乙图像素实验历史目录)`内的真实购买历史，并核验来源文件摘要；每组十五条历史包含十四条正常历史和一条甲商品真实购买历史。原始历史文字、动作、真实购买商品和源图保留，新目录生成频谱模拟攻击图、替换证据、记忆库与检索索引。

流程依次完成历史复制与核验、三组图片优化、记忆替换、检索索引准备、本地预检、商城服务启动和购物评估。准备和索引生成阶段不调用购物模型；新目录不复制旧购物结果，也不复用旧像素攻击图的检索缓存。完整运行按每组相同的二十条模糊同款需求分别测试无记忆、干净记忆、攻击记忆，共 **3 组 × 3 条件 × 20 次 = 180 个购物回合**。完整图文联合检索、完整历史文字、单条记忆展示和固定乙参考图沿用旧实验，备用需求为零。技术错误照实记录，不能把启动成功视为全部回合成功完成。

在 PowerShell(命令行脚本环境)中从项目根目录执行本轮参数：

```powershell
.\run_old_b_frequency_126.ps1 -Stage Run -Method ssa `
  -AttackSteps 500 -SsaSamples 20 -SsaRho 0.5 -SsaSigma 16 -SsaMomentum 1.0 `
  -Root '.\src\output\nine_groups_old_b_ssa_126_500steps_20261006' `
  -Port 3007 -Device cuda
```

旧乙图脚本在不传输出目录时按方法选择独立默认目录：频域动量使用 `src/output/nine_groups_old_b_frequency_126_500steps_20261006(旧乙图频域动量目录)`，频谱模拟使用 `src/output/nine_groups_old_b_ssa_126_500steps_20261006(旧乙图频谱模拟目录)`。修改步数或采样参数测试新实验时，仍应显式选择新的独立目录。

保留的原频域方法可这样选择：

```powershell
.\run_old_b_frequency_126.ps1 -Stage Run -Method frequency_momentum `
  -AttackSteps 500 -FrequencyAxisRatio 0.25 -FrequencyMomentum 1.0 `
  -Root '.\src\output\nine_groups_old_b_frequency_comparison_new'
```

通用[九组实验入口](../run_nine_groups.ps1)、[二十次本地实验入口](../run_local20.ps1)与单组图片记忆准备入口也接受频谱模拟方法及其四项参数。准备、优化、核验和恢复必须保持冻结的方法、步数、有效参数与乙参考图一致。

## 未指定乙图时的默认值

**项目通用九组入口的默认值是旧乙图。** 准备新的实验时，不传 `-ReferenceOverrides(乙商品与参考图覆盖参数)`／`--reference-overrides(底层参考图覆盖参数)`，就采用 `src/data/victim_target_pair/victim_target.json(原始甲乙商品配对文件)`及其原图。第 1、2、6 组的旧乙商品编号分别为 `B08V134QWR(旧相机商品编号)`、`B09CYFD5BX(旧椅子商品编号)`、`B09HH96NJY(旧牛仔裤商品编号)`。

[旧乙图三组脚本](../run_old_b_frequency_126.ps1)固定复用旧计划，所以默认旧图；[新乙图三组脚本](../run_identity_frequency_126.ps1)在脚本内显式传入[新乙图覆盖文件](../src/data/reference_overrides_identity_126.json)，所以运行该专用脚本即使用新图，无需再额外传参数。覆盖文件同时替换乙商品编号与选定参考图片。

| 组别 | 旧乙图文件，位于旧实验的原图目录 | 新乙图原始审阅文件 |
|---|---|---|
| 1，相机 | `src/output/nine_groups_vague_top1_batch_126/assets/5351fe5727bf6d9d8313c8262c28589948c84856f535b31f6f58d4d44d68166b.jpg(旧相机参考图)` | `src/output/identity_clear_b_review_20261005/01_camera/B07JQR37W6_detail.jpg(新相机细节参考图)` |
| 2，椅子 | `src/output/nine_groups_vague_top1_batch_126/assets/8fc2834a77b8ecf235f5bec9c1f335830d1de142843cda13284cdfe36579000d.jpg(旧椅子参考图)` | `src/output/identity_clear_b_review_20261005/02_chair/B08332K9XN_image00.jpg(新椅子参考图)` |
| 6，牛仔裤 | `src/output/nine_groups_vague_top1_batch_126/assets/acf3f5d8944c118d04e2ba456926f4750e87198bbd3201302a34e52e44bd6039.jpg(旧牛仔裤参考图)` | `src/output/identity_clear_b_review_20261005/06_jeans/B07MTH196Z_image05.jpg(新牛仔裤参考图)` |

新乙商品编号分别为 `B07JQR37W6(新相机商品编号)`、`B08332K9XN(新椅子商品编号)`、`B07MTH196Z(新牛仔裤商品编号)`，新图准备时复制进对应独立实验的原图目录。已经准备或冻结的实验以后续输出目录内的 `manifest.json(冻结计划文件)`和历史绑定为准；恢复时省略覆盖参数不会将新图实验改回旧图。

2026 年 10 月 6 日启动前审计确认本地模型运行环境、三代理权重文件、完整商品目录和旧三组历史均已存在，显卡计算可用，未发现占用 3007 端口的其他商城实验。正式模型请求仍依赖已配置购物模型服务的运行状态。图片优化耗时随采样次数增长，建议这轮串行运行三组，避免同一显卡同时启动多个图片优化实验。
