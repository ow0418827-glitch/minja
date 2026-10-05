# Ubuntu（服务器操作系统）运行指南

适用当前严格配对版本。服务器已有 dual-rag（双阶段检索增强生成实验项目）和视觉权重时，复用源码、权重与原显卡运行环境；另建购物服务环境。以下为部署指令，本次未连接远程服务器验证。

## 1. 上传范围与公共路径

需要三部分：当前 `minja` 目录、服务器已有的 `dual-rag` 目录，以及本机相邻 `MINJA-main/rap/webshop` 中的购物服务代码与数据。当前修改目录不包含购物服务。不要把 Windows（本地操作系统）的环境目录复制到服务器。

示例目录：

```text
~/projects/minja/
~/projects/dual-rag/
~/projects/MINJA-main/rap/webshop/
~/experiments/minja-pair-01/
```

以下公共变量需要在每个新终端中设置；也可保存为一个自己的环境变量文件再加载。按实际服务器路径替换。实验目录使用新的目录。

```bash
export MINJA_ROOT="$HOME/projects/minja"
export DUAL_ROOT="$HOME/projects/dual-rag"
export WEBSHOP_ROOT="$HOME/projects/MINJA-main/rap/webshop"
export RUN_ROOT="$HOME/experiments/minja-pair-01"
export PYTHONUTF8=1
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
mkdir -p "$RUN_ROOT"
```

在 VS Code（代码编辑器）里安装 Remote - SSH（远程安全连接扩展），连接服务器后打开 `MINJA_ROOT` 对应目录，选择服务器上的解释器。连接方式见[官方说明](https://code.visualstudio.com/docs/remote/ssh)。

## 2. 购物代理和图片优化环境

用 Conda（环境管理工具）克隆服务器上之前成功运行视觉项目的环境。把下面的 `YOUR_EXISTING_ENV` 替换为真实环境名；新环境命名为 `minja-agent`。

```bash
conda env list
conda create -n minja-agent --clone YOUR_EXISTING_ENV -y
conda activate minja-agent
python --version
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
```

项目代码至少需要 Python（编程语言）3.10；已有 3.11 环境可直接使用。保留已有可工作的 PyTorch（张量计算框架）与 torchvision（视觉工具库）组合。下面先记录这两个包的版本约束，再补代理依赖，依赖冲突会报错而非直接替换它们。

```bash
python - <<'PY'
import importlib.metadata as m
import os
from pathlib import Path
text = ''.join(f'{n}=={m.version(n)}\n' for n in ('torch', 'torchvision'))
Path(os.environ['RUN_ROOT'], 'torch-constraints.txt').write_text(text, encoding='utf-8')
PY

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -c "$RUN_ROOT/torch-constraints.txt" \
  'open_clip_torch==3.3.0' 'sentence-transformers>=3,<6' \
  'openai>=1,<3' requests beautifulsoup4 PyYAML Pillow 'pytest>=8,<10'

cd "$MINJA_ROOT"
python -m pytest src/tests -q
```

本地当前版本为 30 项测试通过；服务器也应跑这套测试。不直接安装 `src/dependencies/requirements.txt`，该文件是范围很大的历史环境清单，包含不属于当前运行路径的包与固定显卡依赖。

### 2.1 复用视觉权重

保持此前运行视觉项目时的模型缓存变量；不要盲目把缓存切换到一个空目录。复制服务器上以前成功使用的优化配置为本次实验配置：

```bash
cp "$DUAL_ROOT/configs/optimization.json" "$RUN_ROOT/optimization.server.json"
export OPT_CONFIG="$RUN_ROOT/optimization.server.json"
```

如果之前使用的是别的优化配置，上面的源路径替换为那一份。配置中的 `surrogates`（代理编码模型列表）默认需要三组权重：

1. `ViT-B-16 / openai`：16 像素分块的基础视觉模型。
2. `ViT-B-32 / openai`：32 像素分块的基础视觉模型。
3. `ViT-B-32 / laion2b_s34b_b79k`：使用另一训练数据来源的 32 像素分块模型。

当前检索使用第二个代理的 CLIP（图文对比编码模型），生成对抗图默认使用全部三个。已有其他 CLIP 或 GME（通用多模态嵌入模型）权重，并不等价于这三组权重齐全；当前流程不需要加载 GME。

自动发现失败时，在本次配置的每个代理条目中填写 `weight_path`（本地权重文件路径），使用服务器绝对路径，例如 `/data/models/.../open_clip_model.safetensors`。不要使用本地盘符路径，也不要把不同加载格式的权重直接改名冒充。可另设 `model_search_roots`（额外模型搜索目录列表）。

```bash
cd "$MINJA_ROOT"
python - <<'PY'
import os
from src.vision.dual_rag_bridge import activate, make_encoder
activate(os.environ['DUAL_ROOT'])
from dual_rag.original_attack import resolve_optimization
cfg = resolve_optimization(os.environ['OPT_CONFIG'])
for item in cfg.surrogates:
    print(item.name, item.pretrained, item.weight_path)
encoder = make_encoder(os.environ['DUAL_ROOT'], os.environ['OPT_CONFIG'], 2, 'cuda')
print('检索编码器加载成功', encoder.device)
PY
```

此检查解析全部权重路径并实际加载第二个模型；后续两步优化再验证全部模型。图片桥接代码禁止自动下载代理权重。准备、排名、购物三个阶段都传入同一个优化配置和代理编号。

### 2.2 准备独立的文字编码模型

代理的动作重排序还需要 `sentence-transformers/all-MiniLM-L6-v2`（轻量句向量模型），它与 CLIP 不同。

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
print(model.encode(['test']).shape)
PY
```

仅在缓存缺失时，通过镜像下载一次；保持与运行时相同的缓存设置：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
PY
```

成功后可在代理终端设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，使后续模型加载使用本地缓存。此设置不提供购物商品图片；商品主图仍须能从其来源地址下载。

## 3. 单独配置购物服务

新环境使用 Python 3.11 与 Java（检索引擎运行环境）21。锁定 Pyserini（检索引擎接口）1.6.0，与本机实际服务版本一致；该版本的官方要求见[发行说明](https://pypi.org/project/pyserini/1.6.0/)。

```bash
conda create -n minja-webshop python=3.11 openjdk=21 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge -y
conda activate minja-webshop

# 服务只做商品检索与评分，安装处理器版本，节省显卡运行库下载。
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'pyserini==1.6.0' 'Flask==2.3.3' cleantext rank_bm25 rich thefuzz \
  'spacy>=3.8,<3.9' requests beautifulsoup4
python -m spacy download en_core_web_sm

export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
java -version
export OPENAI_API_KEY=not-used
python -c 'from pyserini.search.lucene import LuceneSearcher; print("检索依赖加载成功")'
```

`en_core_web_sm`（英语分词模型）若已安装可跳过下载；该模型下载可能需要访问其官方发布站。购物服务里的 `not-used` 只用于绕过未使用的接口编码器在导入时的初始化检查，不能用于代理请求真实模型；两个服务使用不同终端。

### 3.1 数据与索引

需要以下三个文件：

```text
data/items_shuffle.json
data/items_ins_v2.json
data/items_human_ins.json
```

优先上传本机已下载文件，或复用服务器已有数据。缺少数据时，以下命令使用镜像下载；仅对缺少的文件执行，已有文件不要覆盖：

```bash
cd "$WEBSHOP_ROOT"
mkdir -p data
for file in items_shuffle.json items_ins_v2.json items_human_ins.json; do
  if [ ! -s "data/$file" ]; then
    curl -fL --retry 3 -C - \
      "https://hf-mirror.com/datasets/YWZBrandon/webshop-data/resolve/main/$file" \
      -o "data/$file.part" && mv "data/$file.part" "data/$file"
  fi
done
python download_full_data.py
```

最后一个命令会核验已存在的两个大文件的内容摘要。当前下载辅助脚本内部使用官方源，并不会因为设置 `HF_ENDPOINT` 自动改用镜像，所以前面直接给出镜像下载命令。第三个文件没有内置预期摘要，后续加载会检查其格式。

已存在且能够被当前检索版本打开的全量索引可复用。否则在空的 `search_engine/indexes` 目录中构建一次；全量转换和索引比较耗内存、磁盘与时间，应预留充足空间，不使用小样本数据替代正式评估。

```bash
export WEBSHOP_DATA_MODE=full
cd "$WEBSHOP_ROOT/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py
python -m pyserini.index.lucene \
  -collection JsonCollection -input resources -index indexes \
  -generator DefaultLuceneDocumentGenerator -threads 1 \
  -storePositions -storeDocvectors -storeRaw
```

不要在复用的视觉环境内执行旧 `setup.sh`，也不要在 Ubuntu 直接执行 `.ps1` 本地脚本。

## 4. 从真实购物目标中选择乙并生成实验配置

这是正式评估需要的步骤。当前示例乙编号 `B09LQHW84R` 在本机人工目标文件中没有任务，不能直接拿它作为有真实奖励的乙任务。本机的小样本服务还把评分目标固定成甲，也不适合乙需求的最终购买评估。

下面脚本在购物服务环境中运行，复用网站实际的目标生成和排序逻辑。默认选择第一个含牙刷关键词且具有人工目标的乙商品；若要指定某个乙，先设置 `B_ASIN` 为它的真实编号。该编号必须有对应任务。脚本从网页数据读取乙主图，并生成共享配置和任务信息。此步骤在正式启动服务前执行，退出后释放商品数据内存。

```bash
conda activate minja-webshop
export WEBSHOP_DATA_MODE=full
export OPENAI_API_KEY=not-used
cd "$WEBSHOP_ROOT"
python - <<'PY'
import json, os
from pathlib import Path
import requests
from PIL import Image
from web_agent_site import app as shop

run = Path(os.environ['RUN_ROOT'])
pair = json.loads((Path(os.environ['MINJA_ROOT']) / 'src/configs/product_pair.example.json').read_text(encoding='utf-8'))
with shop.app.test_client() as client:
    response = client.get('/pair_setup')
    assert response.status_code == 200, '商品数据或索引初始化失败'
a = pair['purchase_product']['asin']
assert a in shop.product_item_dict, '甲不在当前商品目录中'
wanted = os.environ.get('B_ASIN')
keyword = pair['requested_product']['keyword']
candidates = [(i, g) for i, g in enumerate(shop.goals)
              if g['asin'] != a and keyword.lower() in g['instruction_text'].lower()
              and (not wanted or g['asin'] == wanted)]
assert candidates, '没有匹配的乙购物目标；请调整商品编号或需求关键词'
index, goal = candidates[0]
b = goal['asin']
pair['requested_product'].update(asin=b, name=shop.product_item_dict[b]['Title'])
image_url = shop.product_item_dict[b]['MainImage']
response = requests.get(image_url, timeout=60)
response.raise_for_status()
image_path = run / 'b-download'
image_path.write_bytes(response.content)
with Image.open(image_path) as image:
    extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif'}[image.format]
    image.verify()
image_path = image_path.rename(run / ('b' + extension))
case = {'session': f'fixed_0_{index}', 'requested_asin': b,
        'instruction': goal['instruction_text'], 'image': str(image_path), 'image_url': image_url}
(run / 'pairing.json').write_text(json.dumps(pair, ensure_ascii=False, indent=2), encoding='utf-8')
(run / 'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(case, ensure_ascii=False, indent=2))
PY
```

如果选择其他商品类别，先复制和修改示例配置，并将脚本的配置读取路径改为该文件。生成的 `case.json`（任务信息文件）所使用的会话编号依赖当前商品数据、目标生成规则与排序；后续服务必须使用同一份代码和数据。

## 5. 启动服务，并配置代理模型接口

终端一，保持购物服务运行：

```bash
conda activate minja-webshop
export WEBSHOP_DATA_MODE=full
export OPENAI_API_KEY=not-used
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
cd "$WEBSHOP_ROOT"
python -m web_agent_site.app --log --attrs
```

终端二，使用代理环境，重新设置第 1 节公共路径与 `OPT_CONFIG`：

```bash
conda activate minja-agent
export OPT_CONFIG="$RUN_ROOT/optimization.server.json"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
curl -f http://127.0.0.1:3000/
```

首次访问会触发服务器加载全量数据，响应可能较慢。需要查看网页时，可在编辑器中转发服务器的 3000 端口。

模型接口采用与 OpenAI（模型接口服务商）协议兼容的地址。设置自己的可用服务地址和模型名称；后续默认会向模型发送乙图片，因此选择支持图片输入的模型。

```bash
export MODEL_BASE='https://YOUR_PROVIDER/v1'
export MODEL_NAME='YOUR_VISION_MODEL'
read -rsp '请输入模型密钥: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY
```

只有文字模型时，后续购物命令加 `--text-only-policy`；图片仍参与检索，但策略不直接看图。原脚本与图片入口的参数拼写不同：前者使用 `--model_name`、`--api_base`，后者使用 `--model-name`、`--api-base`。

## 6. 生成严格配对的文字记忆

先用一组注入需求和少量正常任务检查整个流程，保留完整五级诱导模板：

```bash
cd "$MINJA_ROOT"
python -m src.agent.minja \
  --pairing-config "$RUN_ROOT/pairing.json" \
  --provider openai --model_name "$MODEL_NAME" --api_base "$MODEL_BASE" \
  --api_key_env OPENAI_API_KEY \
  --inject_num 1 --num_benign 1 --test_num 1 --num_steps 15 \
  --webshop_file src/data/webshop_instructions.json \
  --memory_output "$RUN_ROOT/memory.json" \
  --test_memory_output "$RUN_ROOT/text_test_memory.json"
```

原程序会调用真实模型，并自动把运行日志写入 `src/output/log/`，终端没有持续输出不一定是卡住。不同实验使用不同输出文件；原程序会初始化这些文件。

查看可以进入图片阶段的记录，优先选择 `inject_` 开头的原始需求阶段记录：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from memory_pairing import load_pairing, validate_paired_memory
run = Path(os.environ['RUN_ROOT'])
pair = load_pairing(run / 'pairing.json')
for row in json.loads((run / 'memory.json').read_text(encoding='utf-8')):
    if not row.get('Pairing'):
        continue
    validate_paired_memory(row, pair)
    image = row.get('PrimaryImage') or {}
    print(row['Id'], row['FinalASIN'], image.get('LocalPath'), image.get('StorageError'))
PY
```

所选记录必须有可读取的主图路径。没有合格记忆时，检查 `memory.json.pairing_attempts.json` 和模型日志，不能直接进入图片准备。严格失败可能来自模型未按模板转向、未买甲、步骤不足或图片下载失败，并不一定是环境问题。

## 7. 获取与真实服务器一致的乙指令

在代理终端读取正式服务的任务页面，避免使用离线生成时不同的价格文案：

```bash
python - <<'PY'
import json, os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
path = Path(os.environ['RUN_ROOT']) / 'case.json'
case = json.loads(path.read_text(encoding='utf-8'))
response = requests.get('http://127.0.0.1:3000/' + case['session'], timeout=600)
response.raise_for_status()
node = BeautifulSoup(response.text, 'html.parser').select_one('#instruction-text')
assert node is not None, '没有找到服务器购物指令'
case['instruction'] = node.get_text(' ', strip=True).split('Instruction:', 1)[1].strip()
path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
print(case['session'], case['requested_asin'], case['instruction'])
PY

export B_SESSION="$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["RUN_ROOT"])/"case.json").read_text())["session"])')"
export B_INSTRUCTION="$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["RUN_ROOT"])/"case.json").read_text())["instruction"])')"
export B_IMAGE="$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["RUN_ROOT"])/"case.json").read_text())["image"])')"
export MEMORY_ID='inject_1'
```

`MEMORY_ID` 必须替换为上一节实际通过检查的编号。文字注入阶段沿用原项目的乙类别需求采样；图像购物评估使用这里明确绑定到乙商品的真实环境会话。

## 8. 图片准备、检索与购物评估

先运行两步验证，检查三模型、图片保存和严格配对；它不能代表正式攻击效果：

```bash
cd "$MINJA_ROOT"
python -m src.experiments.visual_memory_experiment prepare \
  --memory "$RUN_ROOT/memory.json" --memory-id "$MEMORY_ID" \
  --pairing-config "$RUN_ROOT/pairing.json" \
  --target-image "$B_IMAGE" --instruction "$B_INSTRUCTION" \
  --dual-rag-root "$DUAL_ROOT" --optimization-config "$OPT_CONFIG" \
  --surrogate 2 --device cuda --steps 2 \
  --output "$RUN_ROOT/visual-check"
```

通过后，用新的输出目录进行正式 1000 步优化：

```bash
python -m src.experiments.visual_memory_experiment prepare \
  --memory "$RUN_ROOT/memory.json" --memory-id "$MEMORY_ID" \
  --pairing-config "$RUN_ROOT/pairing.json" \
  --target-image "$B_IMAGE" --instruction "$B_INSTRUCTION" \
  --dual-rag-root "$DUAL_ROOT" --optimization-config "$OPT_CONFIG" \
  --surrogate 2 --device cuda --steps 1000 --epsilon-levels 16 \
  --output "$RUN_ROOT/visual-full"

python -m src.experiments.visual_memory_experiment rank \
  --experiment "$RUN_ROOT/visual-full" \
  --dual-rag-root "$DUAL_ROOT" --optimization-config "$OPT_CONFIG" \
  --surrogate 2 --device cuda

python -m src.experiments.visual_memory_experiment shop \
  --experiment "$RUN_ROOT/visual-full" \
  --session "$B_SESSION" --query-image "$B_IMAGE" --instruction "$B_INSTRUCTION" \
  --webshop-url http://127.0.0.1:3000 \
  --provider openai --model-name "$MODEL_NAME" --api-base "$MODEL_BASE" \
  --api-key-env OPENAI_API_KEY \
  --dual-rag-root "$DUAL_ROOT" --optimization-config "$OPT_CONFIG" \
  --surrogate 2 --device cuda --num-steps 15 \
  --output "$RUN_ROOT/shopping-full"
```

已生成的对抗图可通过 `--adversarial-image` 导入，仍会核验与同一甲原图的像素预算。第一轮默认纯图片检索；图文后融合可另建实验，准备时加入 `--key-mode joint --query-mode joint`。默认候选数和保留数均为 1，扩大准备阶段的 `--top-k` 后，购物的 `--candidates` 必须不小于该值。

如果视觉模型不支持图片输入，最后一个命令加入 `--text-only-policy`，其他命令保持一致。

## 9. 结果和常见问题

| 文件/现象 | 如何判断 |
|---|---|
| `memory.json.pairing_attempts.json` | 注入阶段是否真的形成乙需求转向甲的合格轨迹 |
| `visual-full/manifest.json` | 商品配对、模型配置、像素预算和实验数据摘要 |
| `visual-full/retrieval_report.json` | 原图与对抗图条件下的实际召回排名 |
| `shopping-full/shopping_report.json` | 无记忆、原图记忆、对抗图记忆三个条件的最终购买结果 |
| 模型已下载但找不到 | 核对当前用户的缓存目录、代理模型变体及 `weight_path` |
| Java 版本错误 | 在服务环境核对 21 版本及 `JAVA_HOME`，重新打开使用正确环境的终端 |
| 商品主图下载失败 | 查看 `StorageError`，确认服务器能访问商品图片来源；模型离线缓存不能代替商品图片 |
| 配对检查失败 | 查看拒绝原因；不要改写商品编号、完成标记或历史动作来绕过检查 |
| 无新增召回 | 原图可能已经排第一，或带图竞争记忆不足；应增加正常带图记忆并检查基线 |
| 乙会话无目标 | 重新执行真实目标选择；换文字指令不会改变服务器评分目标 |

全量商品数据会占用较多内存，索引也需要额外磁盘空间。资源规模以服务器实际数据加载情况为准；如果显存不足，应先通过两步优化定位，避免静默更换模型或精度后与已有实验比较。

本机已验证代码与 30 项测试，但这些服务器命令尚未在您的 Ubuntu 上执行。运行成功也不等于攻击成功；最终应看实际召回与购买商品，并与原图记忆条件比较。
