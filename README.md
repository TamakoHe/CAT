# 质控云眼---基于华为香橙派云质检一体系统

> 本项目的技术目的聚焦于破解当前工业缺陷检测领域的核心痛点，通过针对性的算法创新与架构设计，实现检测精度、场景适应性与部署效率的全方位提升，为复杂工业环境下的质量控制提供可靠技术支撑。

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()  ![MindSpore](https://img.shields.io/badge/MindSpore-2.7.0-blue.svg)

---

# 目录（Table of Contents）

- [质控云眼---基于华为香橙派云质检一体系统](#质控云眼---基于华为香橙派云质检一体系统)
- [目录（Table of Contents）](#目录table-of-contents)
- [特性（Features）](#特性features)
- [快速开始（Quick Start）](#快速开始quick-start)
- [项目结构（Project Structure）](#项目结构project-structure)
- [数据集（Dataset）](#数据集dataset)
- [配置（Configuration）](#配置configuration)
- [许可（License）](#许可license)

---

# 特性（Features）

- ✅ 支持 CPU / 华为昇腾 NPU训练
- ⚡ 提供训练、验证与推理完整流水线
- 🔧 使用统一TOML配置，简单易用
- 📦 高度模块化，易于维护
- 🧪 有良好的检测精确度和泛化性

---

# 快速开始（Quick Start）

```bash
# 克隆仓库
git clone https://github.com/TamakoHe/CAT.git
cd CAT

# 建议使用 conda
conda create --name cat python=3.11
conda activate cat
# 安装依赖
pip install -r requirements.txt


# 运行
python main.py --config /path/to/config.toml
```

---

# 项目结构（Project Structure）

```
CAT/
├── .gitignore
├── config/
├── datasets/
├── model/
├── utils/
├── main.py
├── README.md
└── requirements.txt

```

---

# 数据集（Dataset）

- kolektorsdd2数据集
  使用的是训练子集被筛选, 只剩下正常样本的数据集(自监督学习)
  链接: https://pan.baidu.com/s/1kI2RksDsoqY1ZVvONkHnhA?pwd=anqf 提取码: anqf
- 自定义数据集
  按照以下结构

```
📂 KolektorSDD2/
├── 📁 ground_truth/      # 缺陷的标注 (掩码) 文件名和对应的测试集文件一致
├── 📁 test/              # 测试集图像
└── 📁 train/             # 训练集图像
```

---

# 配置（Configuration）

- 使用 `config/*.toml` (总体配置文件) 管理模型无关的总体配置。
- 示例 `config/exp1.yaml`：

```toml
[GLOBAL]
RNG_SEED=42
OUTPUT_DIR="./runs"
save_model=true
device.mindspore="Ascend"
device.pytorch="cuda"
[DATASET]
name="KolektorSDD2"
shape.datset=[256,256]
shape.model_input=[256,256]
in_channels=3
path.train="/root/CAT/data/KolektorSDD2"
path.eval="/root/CAT/data/KolektorSDD2"
path.test="/root/CAT/data/KolektorSDD2"
[TRAIN]
enable=false
checkpoint.enable_save=true
checkpoint.save_best=true # 除了save_freq之外 是否单独保存最好  要求EVAL.enable=true
checkpoint.save_freq=-1 #  多少个epoch保存checkpoint -1意味着只在save_best=true的时候保存最后结果
setup.batch_size=256
setup.num_workers=8
setup.learning_rate=0.0005
setup.epochs=200
setup.weight_decay=0.05
setup.warmup_epochs=20
setup.load_pretrain_model=true
setup.pretrain_model_path="/root/CAT/runs/run1/weight/last.mindpt.ckpt"
setup.optimizer.name="adam"
setup.optimizer.beta1=0.9
setup.optimizer.beta2=0.95
[EVAL]
enable=false
setup.batch_size=64
setup.num_workers=10
[TEST]
enable=true
setup.batch_size=64
setup.num_workers=8
visualize.enable=true
visualize.mode="all" # 要么填入all或者一个0到1之间的浮点数表示每个测试被中的概率 
[MODEL]
name="CAT"
framework="mindspore" 
config.class_name="CAT_config"
config.path="/root/CAT/config/CAT.toml"
```

- 使用 `config/<model_name>.toml` 存储模型特有的配置
- 示例 `config/CAT.toml`

```toml
DA_low_limit=0.7
DA_up_limit=1.0
layers_to_extract_from= ["layer1", "layer2", "layer3"]
feature_compression=false 
scale_factors=[4.0, 2.0, 1.0]
FPN_output_dim=[256, 512, 1024]
patch_size=[16,16]
embed_dim=64
depth=12
num_heads=8
mlp_ratio=4
window_size=4
qkv_bias=true
drop_path_rate=0.1
input_resolution=16
pretrain_image_size=[224,224]
backbone="resnet50"
```

<!-- # 模型与结果（Models & Results）
<sub>**Table:** Comparison with Existing Methods on KolektorSDD2 Dataset</sub>

| Method | Image-level AUROC | Pixel-level AUROC | Per-region Overlap | AP_det | AP_loc |
|:--|:--:|:--:|:--:|:--:|:--:|
| SuperSimpleNet [[1]](#ssn1) | 87.60% | 99.10% | 94.49% | 0.82 | 0.41 |
| DRA [[2]](#DRA) | 86.80% | 84.40% | 56.90% | - | - |
| ReContrast [[3]](#recontrast) | 90.86% | 99.11% | 91.95% | **0.82** | 0.46 |
| MMR [[4]](#mmr) | 92.29% | 99.45% | 95.50% | 0.74 | 0.36 |
| RD4AD [[5]](#RD4AD) | 96.00% | 97.60% | 94.70% | 0.44 | - |
| DREAM [[6]](#DREAM) | 81.10% | 85.60% | 67.90% | 0.39 | - |
| PatchCore [[7]](#PatchCore) | 94.60% | 97.10% | 89.30% | 0.50 | - |
| PRN [[8]](#PRN) | 96.40% | 97.60% | 94.90% | 0.73 | - |
| DFR [[9]](#dfr) | **93.80%** | 99.15% | **96.30%** | 0.76 | 0.37 |
| ADClick-Seg [[10]](#ADClick) | 91.50% | 99.50% | - | - | - |
| **Ours** | **93.80%** | **99.54%** | 95.80% | **0.82** | **0.49** |



- 表格展示不同实验对比（建议放在 `docs/` 或 `RESULTS.md`） -->

# 许可（License）

本项目采用 GPLv3。详见 `LICENSE` 文件。
