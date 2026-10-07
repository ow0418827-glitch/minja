# Memory Injection Attacks on LLM Agents via Query-Only Interaction (NeurIPS 2025)
本工作目录是购物代理部分的独立修改副本。实现代码按功能位于 `src/`，文档位于 `docs/`，配置、输入数据、测试和实验结果分别位于 `src/configs/`、`src/data/`、`src/tests/` 和 `src/output/`。购物网站服务与大体积商品数据沿用相邻 `MINJA-main/rap/webshop` 目录。

阅读入口：[目录结构与代码导航](docs/PROJECT_STRUCTURE.md) · [本地 20 次实验运行说明](docs/LOCAL20.md) · [购物代理说明](docs/RAP_GUIDE.md)。

图片对抗优化、长期记忆检索和三条件购物实验见 [图片记忆实验入口](docs/VISUAL_MEMORY.md)。

图像优化新增可选 `frequency_momentum(频域动量增强方法)`，支持运行参数选择频带与动量，原默认像素流程保持不变；方法取舍、运行指令和保存图片验证见 [可选频域图像优化](docs/FREQUENCY_OPTIMIZATION.md)。

九组定向实验默认保存“十四条正常乙历史＋一条乙历史图片替换”共十五条记忆，将完整历史文字与图片联合编码，从中检索最相关的前三条提供给多模态模型。每组正式测试固定指定历史实际购买的乙商品及原图，使用二十种语义一致表述；它们是同一实例的重复测量。中性模板、干净／无记忆对照、完整磁盘商品库与运行命令见 [九组定向图文记忆实验](docs/NINE_GROUPS.md)，入口为 `run_nine_groups.ps1`，默认新输出目录为 `src/output/nine_groups_targeted_joint`，保留旧实验输出。

联合检索默认采用 CLIP(对比语言图像预训练模型)，可选 GME(通用多模态嵌入模型)。两者已完成本地短图文编码实测；CLIP 长输入完整覆盖也有真实模型验证，GME 长历史覆盖目前由单元测试验证。完整九组购物尚未运行。评估结果按模板摘要与检索协议摘要隔离，并记录编码器快照及测试清单摘要，防止恢复时混用。

新增的纯文字检索位于 `src/retrieval/`，启用后为四条件购物实验。代码分工、运行指令与结果解释见[纯文字检索基线](docs/TEXT_BASELINE.md)。

We propose MINJA, a novel memory injection attack that injects malicious records into LLM agents through queries. 
<p align="center">
  <img src="figure/MINJA.png" width="800">
</p>

# Getting Started
We conducted our experiments on three agents: RAP Agent, EHR Agent, and QA Agent.
* [RAP Agent](docs/RAP_GUIDE.md)
* EHR Agent
* QA Agent

# Citation

新增可选 SSA(频谱模拟攻击)，保留原频域扰动指令接口。旧乙图第 1、2、6 组完整三条件实验可通过 `run_old_b_frequency_126.ps1(旧乙图三组实验脚本)`传入 `-Method ssa(选择频谱模拟方法参数)`启动；通用入口未指定参考图覆盖时默认旧乙图，新乙图专用脚本明确固定使用新图。参数、官方来源与完整运行示例见[频谱模拟方法说明](docs/SSA_OPTIMIZATION.md)。

```bash
@inproceedings{dong2025memory,
  title={Memory Injection Attacks on LLM Agents via Query-Only Interaction},
  author={Dong, Shen and Xu, Shaochen and He, Pengfei and Li, Yige and Tang, Jiliang and Liu, Tianming and Liu, Hui and Xiang, Zhen},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
[529 服务过载的延后补跑与断点续跑](docs/529_RECOVERY.md)：收到过载时保留原需求，先继续其他正式需求，再最多补两轮；连续三个需求过载或额度耗尽时保存并退出，支持同目录续跑。
