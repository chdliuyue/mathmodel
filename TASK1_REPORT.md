# 任务1：跨域轴承数据处理与特征构建全流程说明

本说明面向希望快速掌握本仓库“任务1”实现细节的研发人员，覆盖从 MAT 原始振动信号加载、源域样本筛选、分段、特征提取到特征表落地及分析脚本的完整链路。文档结合源码目录（`src/data_io`、`src/feature_engineering`、`src/pipelines`、`src/tasks/task1` 与 `scripts/extract_features.py` 等）逐段拆解，帮助读者在不阅读全部代码的情况下理解每一个数据流和关键参数的作用。

---

## 一、任务定位与总体结构

- **目标**：从源域（台架）与目标域（现场）振动信号中抽取时域、频域、包络域、故障频带以及多模态时频特征，生成可直接用于后续建模（任务2）与迁移（任务3/4）的统一特征表及数据字典。
- **总体流程**：
  1. 读取配置（`config/dataset_config.yaml`）并解析输出路径；
  2. 调用 `src/data_io/mat_loader.py` 读取 MAT 文件并构建 `FileSummary`/`SignalRecord`；
  3. 依据 `SelectionConfig` 在源域内选择与目标工况最相近的文件（`dataset_selection.py`）；
  4. 通过 `segment_signal` 对每个通道滑窗分段，调用 `FeatureExtractor` 叠加多类特征（`feature_engineering/`）；
  5. 由 `FeatureDatasetBuilder` 汇总分段特征，补充元数据并写出 CSV；
  6. 如启用附加脚本（`scripts/analyze_features.py`），进一步生成统计表、时频图、特征翻译字典等分析成果。

---

## 二、核心模块代码地图

| 模块 | 关键类/函数 | 作用概述 |
| --- | --- | --- |
| `src/data_io/mat_loader.py` | `LabelInfo`、`SignalRecord`、`FileSummary`、`load_source_file` | 负责解析 MAT 文件，自动识别常见通道名（DE/FE/BA/SENSOR），推断采样率、转速并拆解文件名中的故障编码、尺寸、负载等信息。 |
| `src/data_io/dataset_selection.py` | `SelectionConfig`、`score_summary`、`select_representative_files` | 计算与目标 RPM/采样率/负载/故障尺寸的差异度，通过加权得分筛选每个故障类别的代表文件。 |
| `src/feature_engineering/segmentation.py` | `segment_signal`、`Segment` | 以窗口+步长滑动的方式切割时序，支持重叠与补齐策略。 |
| `src/feature_engineering/statistics.py` & `spectral.py` | `time_domain_features`、`frequency_domain_features`、`envelope_features`、`fault_frequency_band_features` | 计算多域统计量，封装故障频带能量与占比。
| `src/feature_engineering/bearing.py` | `BearingSpec`、`DEFAULT_BEARINGS` | 定义 SKF6205/6203 等轴承几何参数、计算 BPFO/BPFI/BSF/FTF 理论频率及 ±带宽。 |
| `src/feature_engineering/feature_extractor.py` | `FeatureExtractor` | 汇总各类特征并按 `fault_`、`time_` 等前缀组织列名，兼容未提供轴承参数的默认值。 |
| `src/pipelines/build_feature_dataset.py` | `FeatureDatasetBuilder` | 将分段特征与元数据合并成 DataFrame，并生成源/目标域的特征表与元数据表。 |
| `src/tasks/task1/pipeline.py` | `run_feature_pipeline` | 承上启下：解析 YAML、执行源域筛选+特征提取+CSV 写入，返回 `FeatureExtractionOutputs`。 |
| `scripts/extract_features.py` | `run_pipeline` | 命令行入口，便于批量运行任务1。

---

## 三、数据源兼容性与加载逻辑

### 3.1 源域 MAT 文件解析

1. **命名解析**：
   - 通过正则 `LABEL_PATTERN`/`SIZE_PATTERN` 识别 CWRU 常见的 IR/OR/B/N 编码及故障尺寸（千分之一英寸），遇到 `_1` 或 `1HP` 等格式均可解析。
   - `LOAD_PATTERN` 支持下划线或 `HP` 结尾的负载写法，缺失时返回 `None`，不会阻断后续流程。
2. **通道识别**：
   - `CHANNEL_HINTS` 将包含 “DE”、“FE”、“BA” 的变量映射到驱动端、风扇端、基座通道；如果变量名是单个字母（目标域常见），则回退至 `SENSOR`。
3. **采样率与转速推断**：
   - 优先寻找递增的时间向量计算 `1/mean(diff)`；若不存在，则依据文件名中的 `48k`/`12k` 自动设定，最后回退到配置默认值。
4. **数据结构**：
   - `SignalRecord` 保存单通道波形、采样率、RPM、故障信息；
   - `FileSummary` 汇总同一文件所有通道，附带 `label_info` 与 `metadata`（通道数量等）。

### 3.2 目标域 MAT 文件加载

- 目标域文件通常只有单通道，`load_target_file` 直接保留首个通道，并从配置读取采样率/RPM；
- 当目标数据缺失时，`run_feature_pipeline` 会打印 `INFO` 级别日志而非报错，确保流水线健壮。

---

## 四、源域样本筛选策略

1. `SelectionConfig` 中的 `rpm_target`、`sampling_rate_target`、`prefer_load`、`prefer_fault_sizes` 等来自配置；
2. `score_summary` 计算如下加权得分：
   - RPM、采样率与目标值的归一化差值分别乘以 `rpm_weight`/`sampling_rate_weight`；
   - 负载、故障尺寸的惩罚函数在缺失时返回 1，保证未知工况不会被优先；
   - 得分越低越匹配，`select_representative_files` 在每个标签内按升序选取 `top_k_per_label` 条（`null`/`all` 表示保留全部）。
3. 元数据表 (`source_metadata.csv`) 记录每个文件的得分、负载、尺寸，方便核查筛选结果是否符合预期。

---

## 五、信号分段与特征家族

### 5.1 分段 (`segment_signal`)

- `window_seconds` 默认 1 秒，`overlap` 默认 50%，即步长 = `window_size * (1 - overlap)`；
- `drop_last` 控制是否丢弃尾部不足一窗的数据，必要时可设置为 `false` 并自动零填充。

### 5.2 特征提取 (`FeatureExtractor`)

| 特征前缀 | 来源函数 | 说明 |
| --- | --- | --- |
| `time_` | `time_domain_features` | 均值、标准差、峭度、峰值因子、信噪比等 17 项统计量。 |
| `freq_` | `frequency_domain_features` | 谱质心、主频、谱熵、谱峰值、总能量等；内部自动汉宁窗+`rfft`。 |
| `env_` | `envelope_features` | Hilbert 包络的时域统计 + 包络谱峰值/带宽/熵。 |
| `fault_` | `fault_frequency_band_features` + `BearingSpec.fault_frequencies` | 输出 FTF/BPFO/BPFI/BSF 的理论频率 (`*_frequency`) 及 ±带宽能量/能量占比。 |
| `tf_` | `src/tasks/task3/features.py`（可选） | 若任务3启用，在此阶段即写入 STFT/CWT/Mel 统计，为后续迁移做准备。 |

- 当缺失轴承参数或 RPM 时，`FeatureExtractor` 会以 0 填充 `fault_*` 字段，确保特征列对齐。

---

## 六、特征表构建与列字典

1. `FeatureDatasetBuilder.build` 遍历每个 `SignalRecord`，针对每个滑窗段生成一行字典：
   - **元数据列**：`dataset`、`file_id`、`file_path`、`channel`、`segment_index`、`start_sample`、`end_sample`、`segment_length`、`segment_duration`、`sampling_rate`、`rpm`、`label`、`label_code`、`load_hp`、`fault_size_inch/mm`、`selection_score`；
   - **特征列**：按前缀组织，可通过 `feature_dictionary.py` 自动翻译成中文含义。
2. 源域表默认写入 `artifacts/task1/source_features.csv`，目标域表写入 `target_features.csv`，并配套 `source_selection.csv`（源域筛选元数据）与 `target_metadata.csv`（目标域文件概览）。
3. 特征列数随配置而变：启用三类时频特征后，列数可超过 150，需注意下游模型的特征筛选策略。

---

## 七、配置文件详解 (`config/dataset_config.yaml`)

| 配置段 | 关键字段 | 说明 |
| --- | --- | --- |
| `source` | `root`、`pattern` | 源域数据目录及 glob 模式；支持子目录递归。 |
| | `default_sampling_rate` | 当 MAT 中缺失采样率时使用的默认值。 |
| | `segmentation` | `window_seconds`、`overlap`、`drop_last` 控制滑窗。 |
| | `selection` | 同上所述的相似度加权参数。 |
| | `channel_bearings` | 通道→轴承型号映射（默认 DE→SKF6205 等）。 |
| | `feature` | 是否启用频域/包络/故障带及带宽设置。 |
| `target` | `sampling_rate`、`rpm` | 目标域统一元数据，可覆盖每个文件内记录。 |
| | `channel_bearings` | 目标域传感器对应轴承。 |
| `outputs` | `directory`、文件名映射 | 调整产物存放位置与文件名。 |

> **提示**：可在 `source.pattern` 中限定某些负载/故障组合，以缩小训练集或进行消融实验。

---

## 八、运行脚本与日志

- 命令行执行：
  ```bash
  python scripts/extract_features.py --config config/dataset_config.yaml --output-dir artifacts/task1_custom
  ```
- 运行流程：脚本首先打印配置摘要，随后依次输出“加载源域文件”“筛选完成”“提取目标域特征”等日志节点，帮助监控耗时段。
- 日志级别说明：
  - `INFO`：常规进度、回退提示（如启用 CWT 解析/数值回退）；
  - `WARNING`：文件缺失、某个 MAT 无信号通道、求解失败等；
  - `DEBUG`（需手动开启）：打印更细粒度的窗口参数、特征列数量等。

---

## 九、配套分析脚本 (`scripts/analyze_features.py`)

- 功能概述：
  - 合并源/目标域特征，生成 `feature_statistics.csv`（均值/方差/极值）；
  - 自动绘制 t-SNE、UMAP（依赖 `run_tsne`）、波形+谱图预览、包络谱 3D 面图；
  - 根据 `BearingSpec` 校验理论频率与包络谱峰值误差，输出《故障特征频率验证.csv》；
  - 调用 `feature_dictionary.py` 写出《特征中英文对照表》，覆盖 `time_`、`freq_`、`env_`、`fault_` 乃至 `tf_` 前缀。
- 常用参数：`--preview-seconds` 控制波形展示长度，`--analysis-dir` 覆盖输出目录。

---

## 十、结果核查与质量控制清单

1. **文件完整性**：确认 `artifacts/task1/` 下生成四个 CSV；若为空，检查数据路径或筛选条件是否过严。
2. **特征分布**：使用分析脚本查看 `feature_statistics.csv` 与 t-SNE 图，验证源/目标域分布差异是否符合预期。
3. **故障频率**：比对《故障特征频率验证.csv》中理论值与包络谱峰值，误差过大通常意味着轴承型号或 RPM 填写有误。
4. **日志告警**：若出现“Segment too short for STFT”等提示，可调节 `window_seconds` 或禁用特定特征族。
5. **再现性**：`selection_score`、`segment_index` 等列为后续伪标签追踪提供唯一索引，严禁在下游阶段擅自删除。

---

## 十一、与下游任务的衔接

- **任务2**：直接读取 `source_features.csv`，使用 `label` 作为目标变量，`selection_score` 可用于加权或抽样。
- **任务3**：复用任务2的建模配置，同时在 `_augment_with_time_frequency` 中读取任务1生成的 `tf_*` 列；若需要动态生成时频特征，可保持 YAML 中的一致参数。
- **任务4**：解释性分析依赖任务3输出的伪标签及 `FeatureDictionary`，确保任务1阶段的列名保持稳定。

---

## 十二、常见问题与排查建议

| 问题现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 源域 CSV 为空 | `source.pattern` 未匹配到文件或所有文件被筛选剔除 | 检查目录与 `top_k_per_label`，或暂时关闭筛选（设置为 `null`）。 |
| `fault_*` 全为 0 | RPM/轴承型号缺失 | 在配置中显式填写 `rpm` 或 `channel_bearings`。 |
| 运行报错 “No numeric feature columns” | 读取的 CSV 为空或所有特征列为非数值 | 确认数据是否被正确写入、是否以 UTF-8-SIG 保存；必要时用 Pandas 检查列类型。 |
| STFT/CWT 警告 | 段长过短或 SciPy 版本不含对应函数 | 调整 `window_seconds`、更新 SciPy 或接受内置回退实现。 |

---

通过以上流程，读者可以明确每一步的输入输出、代码位置以及可调参数，从而在本仓库基础上扩展特征、替换数据源或嵌入自定义的诊断策略。
