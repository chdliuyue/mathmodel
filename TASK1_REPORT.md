# 任务1：数据处理全流程与数据字典

## 背景与目标

竞赛的任务1聚焦于“数据分析与故障特征提取”。本仓库已经实现从源域与目标域振动信号中筛选代表样本、分段、计算多域特征并输出结构化特征表的全流程。核心代码位于 `src/data_io`、`src/feature_engineering` 与 `src/pipelines`，并通过脚本 `scripts/extract_features.py` 完成自动化处理。

## 数据资源概览

### 源域（台架试验数据）

- 假定目录：`sourceData/`。虽然仓库未附带 CWRU 原始数据，但 `src/data_io/mat_loader.py` 保留了 Case Western Reserve University (CWRU) 常见命名与变量结构的兼容逻辑：
  - 读取 `.mat` 文件时会优先查找 `DE_time`/`FE_time`/`BA_time` 等典型变量名，同时兼容竞赛提供数据中的自定义通道名称；
  - 文件名解析支持 CWRU 的 IR/OR/B/N 缩写、故障尺寸与负载（0–3HP）模式，在缺少这些字段时会自动回退为“未标注”状态，不会阻断流程；
  - 若 `RPM`、采样率等元数据在文件中缺失，会根据文件名或配置给出的默认值推断，从而与竞赛数据的结构保持一致。
- 文件命名中包含故障类型（IR/OR/B/N）、故障尺寸、负载等级等信息，`src/data_io/mat_loader.py` 负责解析并构造 `FileSummary`/`SignalRecord` 数据结构。该模块同时推断采样率、转速等元数据，并将通道与轴承型号（如 SKF6205/6203）建立映射，方便后续故障频率计算。
- `src/data_io/dataset_selection.py` 根据配置计算与目标工况的相似度，为每个类别挑选 `top_k_per_label` 个代表文件，保证源域数据与目标域尽可能对齐。

### 目标域（现场采集数据）

- 仓库自带 16 条 8 秒振动信号，位于 `targetData/*.mat`。
- 采样率与转速已知（默认 32 kHz、600 rpm），`load_target_directory` 会读取单通道信号并保存成与源域一致的结构，便于统一处理。

## 数据处理流程

1. **源域筛选**：`SelectionConfig` 对 RPM、采样率、负载、故障尺寸分别赋予权重，结合 `score_summary` 函数的归一化差异度计算，为每一类标签选出最相似的文件。
2. **信号分段**：`segment_signal` 使用滑窗（默认 1 s，重叠 50%）将长时序切割成固定长度片段，缓解非平稳影响并扩充样本量。
3. **特征提取**：`FeatureExtractor` 组合多个特征族：
   - `time_domain_features` 计算均值、标准差、峭度、峰值因子等时域统计量；
   - `frequency_domain_features` 获取谱质心、带宽、谱熵、主频、总能量等频域描述；
   - `envelope_features` 通过 Hilbert 包络进一步提取包络统计与包络谱信息；
   - `fault_frequency_band_features` 根据轴承几何参数（SKF6205/6203）计算 BPFO/BPFI/BSF/FTF 附近的能量与能量占比。
4. **结果输出**：`FeatureDatasetBuilder` 为每个分段生成一行特征记录，包含元数据（文件、通道、段索引、采样率、标签、故障尺寸等）以及所有特征字段，最终写入 CSV 供后续建模或分析使用。

### 2025-09-21 功能完善

- **时频增强**：新增 `src/tasks/task3/features.py`，在运行任务3前即可复用，自动计算 STFT 与 CWT 统计量并以 `tf_*` 前缀拼接回特征表，为后续多模态诊断预留接口。
- **故障频率校验**：结合 `BearingSpec.fault_frequencies` 输出的理论值，在特征表中新增 `fault_*_frequency` 字段，同时 `scripts/analyze_features.py` 会生成《故障特征频率验证.csv》，自动比较理论频率、包络谱峰值及误差，形成可追溯的验证记录。
- **特征翻译词典**：`scripts/analyze_features.py` 会联动 `src/analysis/feature_dictionary.py` 输出《特征中英文对照表》，覆盖元数据、时域/频域/包络/故障带以及新增的时频特征，便于撰写报告与课堂展示。
- **可视化优化**：协方差热图改为展示皮尔逊相关性，保留对角线为 1 的直观表现；时间序列网格自动抽取不同文件与通道，避免同一编号（如 B007）刷屏，提升报表的可读性。

> **延伸阅读**：基于该特征表开展的源域诊断建模流程与可解释性分析详见 [`TASK2_REPORT.md`](TASK2_REPORT.md)，可直接复用任务1输出完成任务2需求。

## 特征表字段说明

特征表包含三大类字段：

1. **元数据列**：
   - `dataset`：数据域标识（source/target）；
   - `file_id` / `file_path` / `channel`：原始文件与通道信息；
   - `segment_index`、`start_sample`、`end_sample`、`segment_length`、`segment_duration`：分段后的定位信息；
   - `sampling_rate`、`rpm`：采样率与轴转速；
   - `label`、`label_code`、`load_hp`、`fault_size_inch/mm`：源域标签与故障尺寸、负载等信息（目标域为空值）；
   - `selection_score`：源域样本与目标工况的匹配得分。

2. **时域特征（前缀 `time_`）**：均值、标准差、方差、均方根、峰峰值、峭度、偏度、形状因子、峰值因子、冲击因子、裕度因子、信噪比等。

3. **频域/包络/故障带特征**：
   - `freq_` 前缀：谱质心、频谱分布的偏度/峭度、主频、谱熵、谱峰值、总能量等；
   - `env_` 前缀：Hilbert 包络的时域统计、包络谱峰值及带宽、包络谱熵等；
   - `fault_` 前缀：针对 FTF/BPFO/BPFI/BSF ±bandwidth (默认 ±5 Hz) 的能量与能量占比。

这些特征字段直接来源于 `src/feature_engineering/statistics.py` 与 `src/feature_engineering/spectral.py` 的函数输出，采用固定命名规则便于后续筛选。

## 数据集制作方式

运行 `python scripts/extract_features.py --config config/dataset_config.yaml` 即可按照配置完成以下步骤：

1. 读取源域、目标域目录及分段、特征、筛选等参数；
2. 若源域数据存在，则在 `artifacts/`（或指定目录）下生成：
   - `source_features.csv`：分段特征表；
   - `source_selection.csv`：被选文件及其匹配得分；
3. 生成目标域 `target_features.csv` 与 `target_metadata.csv`；
4. 所有输出均包含完整的元数据与特征列，可直接用于分析或建模。

## 统计分析与可视化

为评估类别可分性、直观理解数据分布，新增脚本 `scripts/analyze_features.py` 支持：

- 读取源/目标特征表并自动合并；
- 计算特征均值、标准差、方差、最小值、最大值等统计量，保存为 `feature_statistics.csv`；
- 生成 t-SNE、UMAP 嵌入图，直观比较源域与目标域特征空间；
- 绘制目标域（及可用时的源域）原始时域波形与谱图诊断视图。
- 自动输出《故障特征频率验证.csv》，对比理论频率、实测峰值频率与幅值，辅助确认轴承参数配置是否正确。

运行示例：

```bash
python scripts/analyze_features.py --config config/dataset_config.yaml --max-records 6 --preview-seconds 3
```

输出默认存放在 `artifacts/analysis/` 目录，可通过参数 `--analysis-dir` 自定义位置。

## 执行流程回顾与结果核对

- **运行流程**：建议首先执行 `python scripts/extract_features.py --config config/dataset_config.yaml`。脚本会在启动时打印配置摘要，并在每个阶段输出进度（已筛选的源域文件数量、分段窗口设置、特征计算状态等），便于核对是否加载了预期的数据目录。
- **结果核查**：流程结束后应在输出目录（默认 `artifacts/task1/`）看到四张 CSV（源/目标特征表与元数据）。其中 `*_metadata.csv` 记录了筛选得分、轴承尺寸、采样率等关键指标，可与 CWRU 原始命名或竞赛数据清单逐项比对。
- **一致性验证**：配合 `scripts/analyze_features.py` 的统计与可视化输出，可以快速审查源域/目标域之间的均值、方差、特征分布差异；若发现异常（如某些频带能量始终为 0），可回溯到 `feature_engineering` 模块定位问题。
- **再利用指引**：任务1生成的 CSV 是任务2-4 的统一输入，只要保持列名一致，即可在不同实验中复用；若更换数据源，仅需调整 YAML 配置并重新运行本流程。

## 成果意义

- **源域筛选**使得后续迁移学习建立在与目标工况更相似的数据上，减少域间差异；
- **多域特征**覆盖时域、频域、包络谱与故障特征频带，兼顾传统诊断指标与机理解释；
- **统一的数据字典**与统计分析脚本便于快速审查数据质量、对比源/目标域差异，为后续模型训练与迁移、自适应处理奠定基础。

如需更多背景与操作说明，可结合 `README.md` 中的流程介绍与本数据字典一并参考。
