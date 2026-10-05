> 本文保留原始运行说明；文中运行目录统一为项目根目录。当前本地实验请优先阅读 [本地运行说明](LOCAL20.md)。

# Get Started (from this directory)
\* MINJA is implemented on top of the [RAP (Retrieval-Augmented Planning)](https://github.com/PanasonicConnect/rap) agent framework and the [WebShop](https://github.com/princeton-nlp/WebShop) environment.

1. Activate the existing environment used for this reproduction
```bash
conda activate minja-vision
```

2. Install the RAP runtime dependencies

```bash
python -m pip install sentence-transformers requests beautifulsoup4 PyYAML scikit-learn matplotlib tiktoken
```

3. Install the WebShop server dependencies (without downgrading the existing
Torch installation in `minja-vision`)
```bash
cd ..\MINJA-main\rap\webshop
python -m pip install "Flask<3" cleantext gdown "gym==0.24.0" pandas pyserini rank_bm25 rich thefuzz tqdm spacy
```



4. For the original OpenAI(开放人工智能接口) setup, create `OpenAI_api_key.txt` under this directory and put the prepared key in it.  Do not commit that file.

# MiniMax(中国版大模型平台) setup

The experiment now also supports MiniMax's OpenAI-compatible endpoint.  The
credential is read from an environment variable and is never written into the
repository or experiment result files.  In PowerShell(Windows 命令行), set it in
the terminal that will launch the experiment:

```powershell
$env:MINIMAX_API_KEY = '<your MiniMax API key>'
```

Use the current MiniMax model name `MiniMax-M2.7` and the `minimax` provider.
The default endpoint is the Chinese mainland endpoint
`https://api.minimaxi.com/v1`; pass `--api_base` only when a different
OpenAI-compatible endpoint is required.

# Setup WebShop

**⚠️ Important:** The upstream `setup.sh` pins Python 3.8 and Torch 1.11, so
do not run it inside `minja-vision`.  The following equivalent, Windows-safe
sequence preserves the chosen environment while using the full catalog.

当前工作副本只包含购物代理代码；本机已准备好的 WebShop 服务和商品数据位于相邻的 `MINJA-main/rap/webshop` 目录。从本目录运行下列服务命令前，先进入该目录：

```powershell
Set-Location ..\MINJA-main\rap\webshop
```

启动后保持服务运行，在另一个终端回到本工作副本的根目录运行代理。代理仍连接本机的 `http://localhost:3000`。需要完整商品数据时，沿用相邻目录已有数据，无需再次下载。

1. Download the complete WebShop data and the reward model.  The data helper
uses a resumable Hugging Face mirror because the upstream Google Drive URLs
often reject automated downloads.

```bash
cd ..\MINJA-main\rap\webshop
python download_full_data.py
python -m spacy download en_core_web_lg
```

2. Build the full Lucene index once.  This is the slowest setup step and needs
roughly 10 GB of free disk beyond the product data.

```bash
powershell -ExecutionPolicy Bypass -File .\build_full_index.ps1
```

3. In a separate terminal, launch the local shop before each experiment:

```bash
cd ..\MINJA-main\rap\webshop
powershell -ExecutionPolicy Bypass -File .\run_webshop_minja.ps1
```

The application listens at `http://localhost:3000`, which is the endpoint
already used by `minja.py`.


# Running MINJA on RAP Agent


MINJA on RAP Agent supports configurable adversarial settings defined by a triple (victim, target, price).
Users can specify these parameters via command-line arguments.
```bash
conda activate minja-vision
```
Example:

```bash
python -m src.agent.minja \
  --provider minimax \
  --model_name MiniMax-M2.7 \
  --victim toothbrush \
  --target "DenTek Professional Oral Care Kit with DenTek Triple Clean Advanced Clean Floss Picks" \
  --target_price 20.0 \
  --inject_num 15 \
  --num_benign 50 \
  --test_num 30 \
  --seed 20260710
```

For an inexpensive smoke test before the full setting, start WebShop with
`run_webshop_minja.ps1 -Smoke` and use the target-preserving smoke instruction
set.  It prevents the reduced catalog from sampling unrelated full-catalog
tasks:

```bash
python -m src.agent.minja --provider minimax --model_name MiniMax-M2.7 \
  --inject_num 1 --num_benign 1 --test_num 1 --num_steps 10 \
  --indication_limit 1 --webshop_file src/data/smoke_instructions.json \
  --memory_output src/output/smoke-memory.json \
  --test_memory_output src/output/smoke-test-memory.json
```

The default full setting follows the paper code: 15 base attack instructions,
50 benign trajectories, and 30 held-out victim instructions.  Results are
written to `src/output/memory.json`, `src/output/test_memory.json`, and a timestamped
`src/output/log/` file.

The complete list of predefined victim–target–target_price configurations is available in:
```bash
victim_target_pair/victim_target.json
```

## 图片长期记忆（第一阶段）

购物轨迹成功写入 `src/output/memory.json` 时，代理会根据最终点击“Buy Now”的商品编号，保存该商品详情页的主图。图片保存在记忆文件同目录的 `memory_images/` 下，以图片内容的 SHA-256 摘要命名；相同图片只保留一份。`src/output/test_memory.json` 使用同一规则。

每条新记忆保留原有的 `Id`、`Instruction`、`Actions`、`Success`、`Reward` 字段，并增加：

- `FinalASIN`：最终购买商品的编号。
- `PrimaryImage`：最终商品主图的元数据，包括 `SourceURL`、相对于记忆文件所在目录的 `LocalPath`、`SHA256`、`Role`、`PageType` 和 `Step`。
- `Images`：此阶段只包含上述主图，预留给以后扩展多图记忆。

例如，`LocalPath` 为 `memory_images/abc123.jpg` 时，可从 `src/output/memory.json` 所在目录解析该路径。外部商品图下载失败时，轨迹仍写入记忆；图像元数据会保留来源地址，`LocalPath` 为 `null`，并附有 `StorageError`。旧记忆缺少这些图片字段也能继续参与现有的文本检索。

上述是图片存储入口。后续图片检索、甲图向乙图的对抗优化、固定文字记忆及三条件购物评估已通过独立实验入口接入，详见 [图片记忆检索实验说明](VISUAL_MEMORY.md)。直接运行原购物脚本时仍采用原文本流程。

文字诱导与对抗图严格配对时，原脚本和图片准备入口使用同一份 `--pairing-config` 商品配置。原脚本据此生成“乙需求→缺货理由→搜索甲→实际购买甲”的记忆；图片入口核验轨迹、最终商品编号及已保存的甲主图后才允许绑定扰动图片。示例配置见 [商品配对配置](../src/configs/product_pair.example.json)，完整步骤及旧记忆兼容限制见上述实验说明。
