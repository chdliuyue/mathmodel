# 高速列车轴承智能故障诊断：任务1-4代码说明

本仓库现已覆盖竞赛题目前四个任务：

* **任务1：数据分析与故障特征提取** —— 自动筛选源域样本、分段并提取多域特征；
* **任务2：源域故障诊断建模** —— 在任务1特征基础上构建具备迁移潜力、且兼顾多层可解释性的诊断模型并输出评估报表。
* **任务3：迁移诊断模型设计** —— 引入时频多模态特征、CORAL 对齐与伪标签策略，对目标域进行分类与标定；
* **任务4：迁移诊断可解释性** —— 从全局/域间/局部三个层面输出迁移模型的解释性报表与图表。

四个任务的结果分别存放于 `artifacts/task1` 至 `artifacts/task4` 目录，图表与表格均以 UTF-8 编码输出，便于直接纳入中文报告。

任务1的核心目标仍然是：

1. 针对源域（台架）数据自动筛选与目标域工况更接近的代表性样本；
2. 结合轴承故障机理提取时域、频域、包络谱以及特征频带能量等特征；
3. 对源域和目标域信号统一输出可用于后续建模的结构化特征表。

目前仓库已经随附了完整的源域 `sourceData/`（Case Western 台架）与目标域 `targetData/`（竞赛提供的 16 条 8 秒信号）。若需要裁剪或替换数据，也可直接在 `config/dataset_config.yaml` 中调整筛选参数，流程会自动适配。

## 目录结构概览

```
mathmodel/
├── config/                # YAML 配置（dataset/task2/task3/task4）
├── scripts/               # 命令行脚本（特征提取/诊断建模/迁移诊断/解释性分析）
├── src/
│   ├── data_io/           # MAT 文件解析与源数据筛选
│   ├── feature_engineering/ # 各类特征计算模块
│   ├── pipelines/         # 组合成特征数据集的流程工具
│   └── tasks/             # 按任务拆分的高层封装（task1-task4）
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

## 任务1：数据分析与故障特征提取

脚本 `scripts/extract_features.py` 读取 `config/dataset_config.yaml` 中的配置完成整个流程。主要步骤如下：

1. **源域数据筛选**（`src/data_io/dataset_selection.py`）：根据轴转速、采样率、故障尺寸及载荷等指标，为每一类别（内圈/外圈/滚动体/正常）自动选取最接近目标工况（600 rpm、32 kHz）的文件；当 `top_k_per_label` 设置为 `null/0/all` 时，将保留该类别下的全部样本（当前默认配置即保留全部 161 个源域样本）。
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

- `source_features.csv` / `source_selection.csv`：源域筛选与特征表；
- `target_features.csv` / `target_metadata.csv`：目标域分段特征与元数据；
- 若通过命令行覆盖了输出路径，也会在对应目录生成同名文件。

## 代码要点

- **轴承几何参数与特征频率计算**：`src/feature_engineering/bearing.py` 中封装了 SKF6205/6203 的几何参数，可扩展至其他轴承。
- **MAT 文件解析**：`src/data_io/mat_loader.py` 可解析带有 `DE`/`FE`/`BA` 信号及 `RPM` 变量的 Case Western 数据，也兼容目标域单变量 `.mat` 文件。
- **可配置性**：全部关键参数（窗口长度、重叠率、筛选策略、输出路径等）均可在 YAML 中调整，方便后续迭代与实验。

## 特征分析与可视化

为了快速评估源/目标域特征的类别可分性并直观理解原始时序信号，新增了分析脚本：

```bash
python scripts/analyze_features.py --config config/dataset_config.yaml --preview-seconds 3
```

默认情况下脚本会一次性展示 **16 条目标域信号** 以及 **19 个源域代表样本**（`48kHz_Normal_data` 全部 4 条 + `12kHz_DE_data`/`12kHz_FE_data`/`48kHz_DE_data` 各自挑选 B、IR 及 OR-`Centered`/`Opposite`/`Orthogonal` 五个代表），对应“4 + 5×3”的目录覆盖策略，确保每个文件夹都能在时序网格图中出现。`--preview-seconds 3` 用于控制时域预览的秒数，仍可按需调节。若需额外限制数量，可使用 `--target-max-records` / `--source-max-records`（或 `--max-records`）覆写默认行为；如需恢复完整样本遍历，可将 `--source-preview-mode` 设为 `diverse`。

脚本会在 `artifacts/analysis/`（或通过 `--analysis-dir` 指定的目录）下生成内容全面的中文可视化/报表：

- `特征统计汇总.csv`：对全部特征列计算均值、标准差、方差、最大值、最小值，并提供中文列名；
- `特征整合表.csv`：融合源域/目标域特征，统一字段命名以便后续任务直接调用；
- `域对齐指标.csv`：输出 MMD、CORAL 等分布一致性指标，评估源目标域差异；
- `特征重要度.png` 与 `特征重要度.csv`：基于随机森林的特征重要度排行；
- `特征协方差热图.png`：展示主要特征之间的皮尔逊相关性；
- `故障特征频率验证.csv`：列出理论频率、包络谱峰值与误差，辅助校验轴承参数设置是否匹配；
- `tsne_embedding.png` / `umap_embedding.png`：源域与目标域特征的低维嵌入图（中文标题与刻度）；
- `target_time_series_overview.png` 与 `target_*`/`source_*` 诊断图：新版时序网格对齐示例图、时频谱图、窗序折线、包络谱、时频显著性热图等多视角信号分析图。
- `*_包络谱.png`：三维包络谱（沿虚拟深度复制包络能量，自动标注理论故障频率，便于旋转观察细节）。

关于任务1数据字典、字段解释及处理流程的完整说明，详见 [`TASK1_REPORT.md`](TASK1_REPORT.md)。

## 任务2：源域故障诊断建模

在完成特征构建后，可直接调用 `scripts/train_task2_model.py` 训练源域诊断模型并生成评估与可解释性结果。模型采用**CORAL 特征对齐 + 标准化 + 带类权重的逻辑回归**结构，默认执行“正常 + 三类故障”的四分类任务，若输入特征表缺少某一预期类别，会在日志中给出提示并在分类报告、混淆矩阵中保留该类别的中文列：

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
| `model_comparison.csv` | 统一划分下的多模型表现对比（准确率、宏 F1 等指标） |

此外，脚本会在同一训练/测试划分下评估随机森林、梯度提升与支持向量机等基线模型，便于与主模型形成量化对比。

详细的建模策略、参数说明与结果解读见 [`TASK2_REPORT.md`](TASK2_REPORT.md)。

## 任务3：迁移诊断模型

`scripts/run_task3_transfer.py` 提供端到端的迁移训练脚本：

```bash
python scripts/run_task3_transfer.py --config config/task3_config.yaml
```

关键特性：

- **多模态特征增强**：自动补充 STFT/CWT 统计量（`tf_stft_*`、`tf_cwt_*`），支持后续多模态建模。
- **CORAL + 伪标签**：复用任务2的 `SourceDiagnosisConfig`，结合目标域均值/协方差完成对齐，按置信度自训练伪标签。
- **对齐诊断**：输出 MMD/CORAL 指标、t-SNE 嵌入（标签颜色 = 源域真实标签 + 目标域预测标签），直观呈现伪标签前后的分类边界。
- **模型存档**：保存 `transfer_model.joblib` 供任务4直接加载解释。
- **可视化补强**：全中文图件，覆盖双轴伪标签演化曲线、结合效应量与相对差值的《多模态特征分布对比.png》、带阈值区域与趋势线的《伪标签一致性散点.png》以及《多模态时频示例.png》，可直接嵌入技术汇报。

详见 [`TASK3_REPORT.md`](TASK3_REPORT.md)。

## 任务4：迁移诊断可解释性

`scripts/run_task4_interpretability.py` 复用与任务3一致的配置，生成全局/域间/局部三个层面的解释性报表：

```bash
python scripts/run_task4_interpretability.py --config config/task4_config.yaml
```

输出包括：

- `global_feature_effects.csv` + `global_importance.png`：多类别系数/赔率比排行及类别聚合，列出中文类别与特征名称。
- `domain_shift_contributions.csv` + `domain_shift.png`：分特征×类别汇总源/目标域均值差异对 logit 的贡献度，采用分组水平条形图直观比较正负影响。
- `local_explanation.csv` + `local_explanation.png`：指定样本的特征贡献排序，带中文目标类别标题与贡献值标注，方便专家核验。
- `interpretability_summary.json`：汇总解释性分析的核心统计。

详细说明参见 [`TASK4_REPORT.md`](TASK4_REPORT.md)。

## 质量自检与快速验证

为确保脚本入口与依赖环境配置正确，建议在拉取或修改代码后执行以下快速检查：

```bash
python -m compileall src scripts              # 校验源码语法是否通过
python scripts/extract_features.py --help     # 确认特征提取脚本可正确加载
python scripts/train_task2_model.py --help    # 确认源域建模脚本入口正常
python scripts/run_task3_transfer.py --help   # 确认迁移诊断脚本参数解析无误
python scripts/run_task4_interpretability.py --help  # 确认可解释性脚本可用
```

上述命令不会真正跑完全部流水线，但能及时发现依赖缺失、路径配置或语法问题。若需完整验证，可结合 `config/` 下的示例 YAML，在本地或服务器执行端到端流程，并对照 `artifacts/` 输出核对结果。

## 下一步建议

- 在现有全流程基础上引入更多轴承类型或不同采样率的数据，验证筛选策略与特征体系的泛化能力；
- 针对任务3 的迁移模型增加在线伪标签更新与概念漂移监控，提升长期部署的稳定性；
- 将任务4 输出的多粒度解释结果整合成交互式仪表盘，便于专家进行快速诊断与回溯。

