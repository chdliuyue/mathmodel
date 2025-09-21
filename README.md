# 高速列车轴承智能故障诊断：任务1代码说明

本仓库实现了竞赛题目第一问（“数据分析与故障特征提取”）所需的主要代码流程。核心目标是：

1. 针对源域（台架）数据自动筛选与目标域工况更接近的代表性样本；
2. 结合轴承故障机理提取时域、频域、包络谱以及特征频带能量等特征；
3. 对源域和目标域信号统一输出可用于后续建模的结构化特征表。

由于题目给出的源域数据体量较大未上传到仓库，代码在设计时兼顾了“数据缺失时不中断”的需求。一旦在本地补充 `sourceData/` 目录即可直接运行完整流程。对于目标域，本仓库已包含 `targetData/` 下的 16 个 8 秒信号，可立即验证特征提取效果。

## 目录结构概览

```
mathmodel/
├── config/                # YAML 配置
├── scripts/               # 命令行脚本
├── src/
│   ├── data_io/           # MAT 文件解析与源数据筛选
│   ├── feature_engineering/ # 各类特征计算模块
│   └── pipelines/         # 组合成特征数据集的流程工具
├── targetData/            # 目标域16个轴承振动数据
├── artifacts/             # 默认的特征输出目录（脚本执行后生成）
├── README.md
└── requirements.txt
```

## 安装依赖

建议使用 Python ≥ 3.10。先创建并激活虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

## 特征提取流程

脚本 `scripts/extract_features.py` 读取 `config/dataset_config.yaml` 中的配置完成整个流程。主要步骤如下：

1. **源域数据筛选**（`src/data_io/dataset_selection.py`）：根据轴转速、采样率、故障尺寸及载荷等指标，为每一类别（内圈/外圈/滚动体/正常）自动选取 `top_k_per_label` 个最接近目标工况（600 rpm、32 kHz）的文件。
2. **信号分段**（`src/feature_engineering/segmentation.py`）：默认将每条 8 秒信号按 1 秒窗口、50% 重叠切分，提高样本数量并缓解长序列非平稳影响。
3. **特征计算**（`src/feature_engineering`）：
   - 时域统计（均值、标准差、峭度、峰值因子等）；
   - 频域特征（谱质心、谱熵、主频、带宽等）；
   - 希尔伯特包络谱特征；
   - 基于轴承几何参数（SKF6205/6203）计算 BPFO、BPFI、BSF、FTF 等特征频率，并提取 ±5 Hz 带宽内的能量与能量占比。
4. **结果保存**：按源域 / 目标域分别输出特征表（`*_features.csv`）及筛选元数据（`*_metadata.csv`）。所有路径可在配置文件中修改。

运行示例：

```bash
python scripts/extract_features.py --config config/dataset_config.yaml
```

运行后，`artifacts/` 目录将包含：

- `target_features.csv`：目标域分段特征（已可生成，因为目标域数据随仓库提供）；
- `target_metadata.csv`：目标域原始文件的概要信息；
- 若在本地放置了 `sourceData/` 并保持结构与 Case Western 数据集一致，还会额外生成 `source_features.csv` 与 `source_selection.csv`。

## 代码要点

- **轴承几何参数与特征频率计算**：`src/feature_engineering/bearing.py` 中封装了 SKF6205/6203 的几何参数，可扩展至其他轴承。
- **MAT 文件解析**：`src/data_io/mat_loader.py` 可解析带有 `DE`/`FE`/`BA` 信号及 `RPM` 变量的 Case Western 数据，也兼容目标域单变量 `.mat` 文件。
- **可配置性**：全部关键参数（窗口长度、重叠率、筛选策略、输出路径等）均可在 YAML 中调整，方便后续迭代与实验。

## 下一步建议

- 对源域特征运行可视化（t-SNE/UMAP）以评估类别可分性，并与目标域特征对比；
- 在 `source_features.csv` 上训练诊断模型（例如梯度提升、1D CNN 等），为任务2做准备；
- 探索统计对齐或对抗迁移等策略，将源域知识迁移至目标域，完成任务3；
- 针对迁移模型输出进一步构建可解释性分析（任务4）。

