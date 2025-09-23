# 任务3：迁移诊断流水线与伪标签策略深度解析

本指南面向需要理解任务3（跨域迁移诊断）全流程的开发者，结合 `src/tasks/task3` 模块、`config/task3_config.yaml` 与 `scripts/run_task3_transfer.py` 等源码，对时频特征增强、CORAL 对齐、伪标签自训练、域差异量化及可视化产物逐一拆解，帮助快速复现并扩展迁移策略。

---

## 一、整体架构回顾

```
源域特征 (task1) ─┐
                  ├─▶ 任务2管线 (逻辑回归 + CORAL)
目标域特征 (task1) ─┘                    │
                                           ├─▶ 伪标签循环 (置信度 + 一致性)
                                           │
                                           └─▶ 可视化/指标输出 + 模型持久化
```

- **重用模块**：直接调用任务2的 `SourceDiagnosisConfig`、Pipeline 与训练逻辑，保证特征处理流程一致；
- **能力扩展**：在源/目标特征表基础上新增 STFT/CWT/Mel 多模态特征、计算域差异指标、生成伪标签诊断图表。

---

## 二、核心代码组件

| 模块 | 关键类/函数 | 功能 |
| --- | --- | --- |
| `src/tasks/task3/configuration.py` | `parse_transfer_config` | 将 YAML 转换为 `TransferConfig`，复用任务2配置并解析时频/伪标签参数。 |
| `src/tasks/task3/features.py` | `TimeFrequencyConfig`、`extract_time_frequency_features`、`compute_multimodal_representation` | 计算 STFT、CWT、Mel 统计以及多模态互信息一致性特征，并提供绘图所需的矩阵表示。 |
| `src/tasks/task3/segment_loader.py` | `SegmentFetcher` | 根据特征表中的 `file_path`、`start_sample`、`end_sample` 懒加载原始时序片段，缓存 MAT 通道以避免重复 IO。 |
| `src/tasks/task3/transfer.py` | `TransferConfig`、`PseudoLabelConfig`、`run_transfer_learning` | 迁移主流程：时频增强、源域模型训练、CORAL 对齐、伪标签循环、域对齐评估与产物封装。 |
| `scripts/run_task3_transfer.py` | - | CLI 入口，负责读取配置、调用 `run_transfer_learning`、写出 CSV/PNG/NPZ、绘制 t-SNE 等。 |

---

## 三、时频特征增强机制

### 3.1 STFT 特征

- 使用 `scipy.signal.stft`（自动选择汉宁窗、`nperseg`、`noverlap`）计算复杂谱；
- 统计量包括总能量、熵、频率轴的均值/标准差/偏度/峭度、时间轴重心、谱峰频率与时间；
- 输出字段命名为 `tf_stft_*`。

### 3.2 连续小波特征

- 支持 `ricker`、`morlet` 等小波，若环境缺失 `signal.cwt` 则自动切换至解析/数值回退实现；
- 统计能量熵、尺度轴统计、能量脊线总能量与平均尺度，字段前缀 `tf_cwt_*`；
- 通过 `_fallback_cwt` 保证在 SciPy 低版本下依旧可运行。

### 3.3 梅尔滤波特征

- 基于 STFT 幅值构建梅尔滤波器组，支持对数幅值与指定频率范围；
- 统计频带均值、标准差、偏度、峭度、时间重心、峰值频率/时间及对比度，字段前缀 `tf_mel_*`。

### 3.4 多模态一致性

- `_compute_consistency_features` 将时域能量包络与 STFT/CWT/Mel 能量沿时间轴的分布进行互信息计算；
- 输出 `tf_consistency_*` 列，衡量模态间特征是否一致，供伪标签筛选参考。

> **注意**：若段长过短或波形缺失，特征会回退为 0 并记录在日志中。

---

## 四、迁移主流程 (`run_transfer_learning`)

1. **时频特征拼接**：
   - 利用 `SegmentFetcher` 从原始 MAT 取出每段时序，计算 `tf_*`、`tf_consistency_*` 并与原特征表按列拼接；
   - 若某段无法取样（文件缺失、索引越界），对应行填空字典并记录缺失数量。
2. **源域模型训练**：
   - 调用任务2的 `train_source_domain_model`，得到 `TrainingResult` 与初始 Pipeline；
   - `feature_columns` 中会自动包含 `tf_*` 特征，保留 CORAL 对齐能力。
3. **目标域对齐与初始预测**：
   - 通过 `_set_target_statistics` 将目标域均值/协方差注入 CORAL；
   - `_predict_with_pipeline` 输出初始预测 (`target_predictions_initial.csv`)，包含概率与指定元数据列。
4. **伪标签循环**：
   - `_apply_pseudo_labelling` 每轮根据 `confidence_threshold`、`consistency_threshold` 选择高置信度段；
   - 支持 `max_iterations` 与 `max_ratio` 限制伪标签数量，避免过拟合；
   - 每轮都会重新克隆 Pipeline、合并伪标签重新训练，并更新 CORAL 目标统计。
5. **域对齐评估**：
   - `compute_domain_alignment_metrics` 计算对齐前/后 MMD、CORAL 距离等指标，并输出 `alignment_before/after.csv`；
   - 同时生成 t-SNE (`tsne_before.png`, `tsne_after.png`)，比较源/目标域嵌入。
6. **产物汇总**：
   - 返回 `TransferResult`，包含源/目标增强特征、伪标签记录、最终预测、时间频率特征列表、一致性特征列表等。

---

## 五、伪标签质量控制

### 5.1 一致性度量

- `_identify_modal_feature_groups` 根据特征前缀将列划分为时域、STFT、CWT、Mel 四组；
- `_evaluate_multimodal_consistency` 对目标样本执行近邻投票，比较模型预测与各模态投票是否一致，并计算均值作为 `consistency_score`；
- 伪标签只有同时满足概率与一致性阈值才会被采纳，相关细节保存在 `pseudo_label_quality.csv`。

### 5.2 产物说明

| 文件/图表 | 说明 |
| --- | --- |
| `pseudo_labels.csv` | 被采纳的目标段（包含预测标签、置信度、一致性得分、伪标签轮次）。 |
| `pseudo_history.csv` | 每轮新增/累计伪标签数、平均置信度/一致性等统计。 |
| `伪标签演化曲线.png` | 双轴展示新增 vs 累计样本数，并叠加置信度/一致性曲线。 |
| `伪标签一致性散点.png` | 以颜色区分伪标签轮次，展示 `max_probability` vs `consistency_score`，星号标记已采纳样本。 |
| `伪标签一致性明细.csv` | （配置中的 `pseudo_quality`）与散点图一一对应，方便人工审查。 |

---

## 六、域间差异与多模态可视化

1. **域对齐表**：`combined_features_before/aligned.csv` 合并源/目标特征，并附带 `dataset`、`label` 列便于后续绘制；
2. **指标文件**：`metrics.json` 记录基础训练指标、伪标签数量、最终特征维度等关键信息；
3. **多模态示例**：
   - `多模态特征分布对比.png` 对 `tf_*` 关键特征绘制均值±标准差条形图，突出源/目标差异；
   - `多模态时频示例.png` + `*.npz`/`*.json` 保存置信度最高样本的波形、STFT、CWT、Mel 矩阵，可直接用于报告；
   - `time_frequency_features.txt` 记录所有新增时频特征名称，便于下游筛选。

---

## 七、配置文件关键项 (`config/task3_config.yaml`)

| 配置段 | 字段 | 说明 |
| --- | --- | --- |
| `features` | `source_table`、`target_table` | 任务1输出路径；`metadata_columns` 控制预测结果中保留的元数据列。 |
| `modeling` | 同任务2 | 复用逻辑回归 + CORAL 配置，可按需调整。 |
| `time_frequency` | `stft`、`cwt`、`mel`、`consistency_bins` | 控制时频特征的窗口、尺度、滤波器数量、互信息直方图分箱。 |
| `pseudo_label` | `enabled`、`confidence_threshold`、`max_iterations`、`max_ratio`、`consistency_threshold` | 伪标签策略。 |
| `outputs` | 各种 CSV/PNG/NPZ 文件名 | 可按需重命名或指定目录。 |

---

## 八、运行流程与日志观察

```bash
python scripts/run_task3_transfer.py \
    --config config/task3_config.yaml \
    --source-features artifacts/task1/source_features.csv \
    --target-features artifacts/task1/target_features.csv \
    --output-dir artifacts/task3_experiment
```

- 关键日志：
  - `INFO`：时频增强完成、源域模型训练、伪标签轮次、对齐指标写入；
  - `WARNING`：段数据缺失、CWT/STFT 回退、伪标签数量超限等；
  - `DEBUG`（手动开启）：打印每轮伪标签选中的索引与特征数量。

---

## 九、结果解读建议

1. **伪标签收敛**：观察 `伪标签演化曲线.png`，若前几轮迅速饱和且置信度稳定，说明阈值合理；反之可调高阈值或减少迭代。
2. **一致性筛查**：结合 `伪标签一致性散点.png` 与 `伪标签一致性明细.csv`，确认被采纳样本分布是否集中在某些通道/文件，避免偏差。
3. **域对齐效果**：比较 `alignment_before/after.csv` 中 MMD、CORAL 距离是否下降；若无改善，可增补时频特征或尝试更深度的对齐方法。
4. **目标预测分布**：对比 `target_predictions_initial.csv` 与 `target_predictions.csv`，评估伪标签是否改善了类别分布与置信度。
5. **多模态特征差异**：阅读 `多模态特征分布对比.png`，识别目标域在特定模态上的能量偏移，为下一轮数据收集或特征工程提供依据。

---

## 十、常见问题排查

| 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `Segment too short for STFT` | 滑窗长度过小 | 增大 `window_seconds` 或禁用时频特征。 |
| 伪标签数量始终为 0 | 阈值过高或一致性不足 | 降低 `confidence_threshold`、`consistency_threshold`，或放宽 `max_ratio`。 |
| `target_predictions.csv` 概率分布异常 | CORAL 对齐未更新 | 检查 `_set_target_statistics` 是否被调用，或调大 `epsilon`。 |
| `alignment_after` 指标比 `alignment_before` 更差 | 伪标签引入噪声 | 调整伪标签轮次、限制最大比例或手动过滤 `pseudo_labels.csv`。 |

---

通过本指南，读者可快速理解任务3迁移诊断的实现要点，并在此基础上调整时频参数、伪标签策略或域对齐手段，构建更适合自身数据的迁移流程。
