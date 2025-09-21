# 高速列车轴承智能故障诊断：任务1&2代码说明

本仓库现已覆盖竞赛题目前两个任务：

* **任务1：数据分析与故障特征提取** —— 自动筛选源域样本、分段并提取多域特征；
* **任务2：源域故障诊断建模** —— 在任务1特征基础上构建具备迁移潜力、且兼顾多层可解释性的诊断模型并输出评估报表。

任务1的核心目标仍然是：

1. 针对源域（台架）数据自动筛选与目标域工况更接近的代表性样本；
2. 结合轴承故障机理提取时域、频域、包络谱以及特征频带能量等特征；
3. 对源域和目标域信号统一输出可用于后续建模的结构化特征表。

由于题目给出的源域数据体量较大未上传到仓库，代码在设计时兼顾了“数据缺失时不中断”的需求。一旦在本地补充 `sourceData/` 目录即可直接运行完整流程。对于目标域，本仓库已包含 `targetData/` 下的 16 个 8 秒信号，可立即验证特征提取效果。

## 目录结构概览

```
mathmodel/
├── config/                # YAML 配置
├── scripts/               # 命令行脚本（特征提取 & 诊断建模）
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
4. **结果保存**：按源域 / 目标域分别输出特征表（`*_features.csv`）及筛选元数据（`*_metadata.csv`）。所有 CSV 均以 `UTF-8-SIG` 编码写出，方便在中文环境中直接查看。

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

## 特征分析与可视化

为了快速评估源/目标域特征的类别可分性并直观理解原始时序信号，新增了分析脚本：

```bash
python scripts/analyze_features.py --config config/dataset_config.yaml --max-records 6 --preview-seconds 3
```

脚本会在 `artifacts/analysis/`（或通过 `--analysis-dir` 指定的目录）下生成内容全面的中文可视化/报表：

- `特征统计汇总.csv`：对全部特征列计算均值、标准差、方差、最大值、最小值，并提供中文列名；
- `特征整合表.csv`：融合源域/目标域特征，统一字段命名以便后续任务直接调用；
- `域对齐指标.csv`：输出 MMD、CORAL 等分布一致性指标，评估源目标域差异；
- `特征重要度.png` 与 `特征重要度.csv`：基于随机森林的特征重要度排行；
- `特征协方差热图.png`：展示主要特征之间的协方差关系；
- `tsne_embedding.png` / `umap_embedding.png`：源域与目标域特征的低维嵌入图（中文标题与刻度）；
- `target_time_series_overview.png` 与 `target_*`/`source_*` 诊断图：新版时序网格对齐示例图、时频谱图、窗序折线、包络谱、时频显著性热图等多视角信号分析图。

关于任务1数据字典、字段解释及处理流程的完整说明，详见新增文档 [`TASK1_REPORT.md`](TASK1_REPORT.md)。

## 任务2：源域故障诊断建模

在完成特征构建后，可直接调用 `scripts/train_task2_model.py` 训练源域诊断模型并生成评估与可解释性结果。模型采用**CORAL 特征对齐 + 标准化 + 带类权重的逻辑回归**结构：

1. **事前可解释性**：模型仅使用任务1已定义的统计/频谱/包络特征，且在训练前自动剔除近乎常数的列；
2. **迁移过程可解释性**：通过 `CoralAligner` 将源域特征白化，可在后续任务中注入目标域统计量完成快速迁移；
3. **事后可解释性**：自动导出逻辑回归系数、赔率比、交叉验证表现以及基于 permutation importance 的特征贡献度。

运行示例：

```bash
python scripts/train_task2_model.py --config config/task2_config.yaml
```

脚本会读取 `artifacts/source_features.csv`（若缺失将提示先执行任务1）并在 `artifacts/task2/` 下生成以下成果：

| 文件 | 说明 |
| --- | --- |
| `metrics.json` | 训练/测试准确率、宏平均 F1、交叉验证均值与方差、CORAL 对齐诊断指标 |
| `classification_report.csv` | `sklearn` 分类报告（含每类 Precision/Recall/F1） |
| `confusion_matrix.csv` | 基于真实标签顺序的混淆矩阵 |
| `coefficient_importance.csv` | 逻辑回归系数与赔率比，便于定量解释各特征贡献 |
| `permutation_importance.csv` | permutation importance 排名（若启用） |
| `predictions.csv` | 测试集预测结果与各类别概率 |
| `feature_summary.csv` | 参与建模特征的统计摘要（均值/标准差等） |
| `features_used.txt` | 训练实际使用的特征清单 |
| `source_domain_model.joblib` | 训练完成的 scikit-learn 管线，可直接加载复用 |

详细的建模策略、参数说明与结果解读见 [`TASK2_REPORT.md`](TASK2_REPORT.md)。

## 下一步建议

- 对源域特征运行可视化（t-SNE/UMAP）以评估类别可分性，并与目标域特征对比；
- 在任务2已有模型基础上进一步探索多源数据/多模型集成，为任务3 的跨域迁移奠定基础；
- 探索统计对齐或对抗迁移等策略，将源域知识迁移至目标域，完成任务3；
- 针对迁移模型输出进一步构建可解释性分析（任务4）。

