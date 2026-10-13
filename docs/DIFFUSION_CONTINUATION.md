# 第七组扩散图片优化续做记录

最新状态：追加的早停及最大日程实验也已结束，500个实际引导步全部完成；最佳第500步，最终统一目标余弦0.893923。175步早停保留50步0.741321，被校准判定误停；最后窗口仍有收益，不能称收敛。全部曲线、失真、模糊压缩与旧方法比较已整理在 src/output/diffusion_group7_early_stop_max_20261011/RESULTS.md，真实最大图为 attack/audit_best.png，不能把早停的 adversarial.png 当最大图。无需继续运行或监听，原背景进程已结束。

最新追加任务：用户要求给扩散方法加入窗口收益早停，并测试最大优化量。该追加工作正在进行，不能将下面已完成的两次旧实验视为新任务完成。设计及配置见 `docs/DIFFUSION_EARLY_STOP.md` 和 `src/configs/diffusion_group7_early_stop_max.json`。真实长实验已在隐藏后台启动，根启动编号48888以实际进程查询为准；输出 `src/output/diffusion_group7_early_stop_max_20261011`，日志 `pipeline.stdout.log` 和 `pipeline.stderr.log`，生成过程实时状态在 `attack/progress.json`，完整候选曲线在 `attack/checkpoint_history.json`。完整日程1000、强度0.5、最多500实际引导，100步后建立基点、每25步检查、连续3窗口增量不足0.002停止，启用停止后完整审计以检查误停。模块默认关闭早停保留旧功能。核心9项停止与审计测试和独立只读审查均通过，生产模块已经冻结以避免运行时更改。不得重复启动已有运行进程。完成后运行 `src.experiments.summarize_diffusion_early_stop` 的实际汇总，统一比较早停图、校准最佳、校准最终和旧30步图，检查全部真实图并向公子交付。临时布局图不可当结果。

状态：工作完成，已实际生成两种配置并完成最终统一比较、目视检查及165项测试；最终结果目录 `src/output/diffusion_group7_20261011_final`。较强配置真实图 `src/output/diffusion_group7_20261011_stronger/attack/adversarial.png`，对目标余弦0.771706，相对原图0.672269提高约0.099437；相对同初值无引导重建0.702701提高约0.069005。原失败和较轻配置均保留。北京时间19点续做任务已执行。无需重复下载或运行。

工作目录：`D:\Material\Literature Reading\code\minja`。已有大量用户未提交修改，不能回滚或覆盖。

独立运行环境：`tmp\diffusion-runtime\Scripts\python.exe`，基于 `D:\Anaconda\envs\minja-vision\python.exe` 的共享原包虚拟环境，只在独立环境安装扩散依赖。模型下载程序：`src\experiments\download_diffusion_model.py`，优先大陆镜像，模型目录 `tmp\models\stable-diffusion-v1-5`。它保存真实模型版本。

第七组源图：`src\output\nine_groups_attacker_history_joint\assets\9b5f4a94d9ab7f2e001498260243b3d91eedcd774939a4558c4f5786192e3a6a.jpg`，多芬泡沫洗手液，商品编号 B08XYBS678。

目标图：同目录 `dd0abf8d2d52aa289346aa2e7219187cec901c901d4f3cce1ef3379691467256.jpg`，洗发水，商品编号 B00BDJODWI。

旧图比较对象：

- 原像素1000次：`src/output/nine_groups_attacker_history_joint/groups/07_shampoo/attack/adversarial.png`。
- 频域动量500次：`src/output/nine_groups_frequency_group7_500steps_20261005/groups/07_shampoo/attack/adversarial.png`。
- 频谱模拟300次原图：`src/output/scar_shampoo_core_20261008/groups/07_shampoo/attack/adversarial.png`。
- 精修后幅度0.625：`src/output/scar_shampoo_old_b_equal05_full_20261008/groups/07_shampoo/attack/adversarial.png`。

现有统一重编码报告：`src/output/lucid_defense_retrieval_20261011/results/summary.csv`。目标图余弦：原图0.6722689867、原像素0.9855914116、频域动量0.8297481537、缩幅精修0.8662887216。采用项目实际检索代理2的单精度模型，入口 `src.vision.dual_rag_bridge.make_encoder(surrogate=2,device='cuda')`。新图与旧图必须在同次运行重算，不能混用训练损失。

实现方案：参考 AdvDiffVLM 论文核心及官方扩散库的 clip_guided_stable_diffusion.py，用真实训练好的稳定扩散1.5去噪器与自编码器，确定性反演或加噪初始化，逆向扩散内注入目标图像相似度梯度及原图重建正则。三代理全局加0.2局部匹配，逐代理图像求梯度，经过可微解码器反传到潜变量。默认256分辨率以适配8吉字节显存。模块 `src/vision/diffusion_attack.py` 正由协作智能体实现，禁止重复覆盖。

事实边界：这是适配本项目的扩散引导实现，并非完整论文基准复现。官方仓库主程序有合并冲突、机器绝对路径和论文与实现的权重差异。确定性反演非无损，自编码器有重建损失，扩散先验不保证严格处于真实图像流形。扩散方法没有16灰度级像素预算，因此报告实际像素差和原图保真度。不能声称必然比旧法更强。

已完成：新模块 `src/vision/diffusion_attack.py`，统一接口 `frequency_attack.generate_attack(method='diffusion')`，独立优化和统一比较入口，原方法相关121项测试通过，新模块与比较26项测试通过。扩散模块另有分派测试已由实现智能体验证。依赖实际安装版本为扩散库0.39.0、加速库1.15.0；原环境包未修改。

下载改为大陆镜像分段下载，支持重用完整分段和最终权重大小及内容摘要校验。下载程序在隐藏后台运行，进程启动编号43780（实际虚拟环境可能有子进程），12个分段并发。日志在 `src/output/diffusion_group7_20261011/download.stdout.log` 和 `download.stderr.log`。检查 `tmp/models/stable-diffusion-v1-5/**/*.parts` 文件增长以及最终 `download_provenance.json`。不能同时重复启动下载。

串行续跑程序：`tmp/diffusion-runtime/Scripts/python.exe -X utf8 -u -m src.experiments.run_diffusion_group7 --wait-for-download`。该程序将等待下载清单完成，实际生成图片并立即统一比较；其阶段状态在 `src/output/diffusion_group7_20261011/pipeline.json`，日志 `pipeline.stdout.log`、`pipeline.stderr.log`。检查是否已有此程序进程，避免重复运行。如失败，先读取异常，修复再用新的空攻击或比较目录；不能覆盖原结果。

实验配置：`src/configs/diffusion_group7.json`，分辨率256，完整日程30步、实际反演和去噪各15步，改图强度0.5，目标引导5，单步潜变量最大位移0.15，原图重建正则1，种子42。实际结果应记录完整元数据。允许按视觉质量和目标效果再调参，但不能假称这些初始参数最优。

更强配置已完成，独立目录 `src/output/diffusion_group7_20261011_stronger`。完整日程60步、实际30步引导，强度0.5，潜变量位移上限0.25，原图正则2，目标引导8。配置 `src/configs/diffusion_group7_stronger.json`。生成约361.65秒，峰值张量显存约5.11吉字节。

结果解释已写入 `docs/DIFFUSION_OPTIMIZATION.md`：两种扩散配置均未超过旧方法未处理相似度；减少明显彩色网格但损失商标文字细节。较强配置模糊后0.703702，仅略高于模糊原图0.701370；压缩后0.758452，低于像素0.769464和频谱模拟0.820384。最大像素差255，原16灰度级预算不满足；属于共享优化模型下的单组适配实验。完整最终报告、数值表和拼图在最终结果目录。`tmp/diffusion_compare_layout_20261011` 为排版测试占位图，绝不能当作真实结果。
