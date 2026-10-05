# 目录结构与代码导航

项目统一采用一个 `src`（源码与实验资源目录），不再保留根目录或源码下的旧同名包目录。

```text
minja/
├── src/
│   ├── agent/          # 购物智能体及 prompts 提示词
│   ├── memory/         # 图片长期存储与商品配对检查
│   ├── retrieval/      # 纯文字指令检索，与图片检索分开
│   ├── vision/         # 对抗优化适配及图片向量检索
│   ├── experiments/    # 数据准备、记忆训练、三／四条件购物实验
│   ├── environment/    # 本地模拟商店
│   ├── configuration/  # 模型配置读取代码
│   ├── configs/        # 模型密钥、商品对和运行参数配置文件
│   ├── data/           # 购物指令、诱导模板、商品对输入数据
│   ├── output/         # 数据子集、索引、图片、训练记录和实验结果
│   ├── tests/          # 回归测试
│   ├── dependencies/   # 依赖清单
│   └── paths.py        # 集中定位项目及资源路径
├── docs/               # 使用与方案说明
├── run_local20.ps1      # 本地实验启动入口
└── README.md           # 阅读入口
```

## 阅读顺序

1. [本地实验运行说明](LOCAL20.md)：当前模型配置、启动方法和结果位置。
2. [图片记忆实验说明](VISUAL_MEMORY.md)：图片存储、检索和三条件对照。
3. [远程运行说明](UBUNTU_RUNBOOK.md)：服务器及模型缓存配置。
4. [原购物代理说明](RAP_GUIDE.md)：传统脚本的历史说明。
5. [纯文字检索基线](TEXT_BASELINE.md)：新增文字检索模块、四条件运行与结果阅读。

代码主线：`experiments/local_experiment.py` 调用 `agent/minja.py` 生成轨迹，使用 `memory/` 保存图片并验证配对，再调用 `experiments/visual_memory_experiment.py` 与 `vision/` 完成图片优化和购物评估。

`configuration/` 保存读取配置的代码，`configs/` 保存实际配置文件，两者职责不同。

## 启动与迁移

根目录的启动脚本名称和参数不变，仍使用 `-AttackSteps 1000` 进行完整优化。旧的底层模块路径不再保留，直接调用时从项目根目录使用：

```powershell
python -m src.experiments.local_experiment --help
python -m src.experiments.visual_memory_experiment --help
python -m src.environment.local_webshop --help
python -m pytest src/tests -q
```

配置从旧目录迁入 `src/configs/`，输入数据迁入 `src/data/`，完整输出目录迁入 `src/output/`。图片与记忆之间的相对路径保持不变。已有实验文件逐个核对迁移前后内容一致；历史诊断记录中的旧绝对输出路径保留为历史记录，不作为新运行路径。

传统智能体入口改为从项目根目录运行 `python -m src.agent.minja`。其默认指令、模板、输出和合并指令路径由统一路径模块定位；显式传入的相对路径仍以当前工作目录为准。

此次仅调整布局与资源定位，模型调用、图片优化、记忆配对、检索、奖励和购买决策逻辑不变。
