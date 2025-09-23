# 任务2：源域故障诊断建模与解释全景指南

本文系统性整理“任务2”中源域模型训练的代码结构、配置项、执行流程与产物说明，帮助读者快速理解如何在任务1特征表基础上构建可迁移且具备多层可解释性的诊断模型。内容覆盖 `src/modeling/task2.py`、`src/tasks/task2/pipeline.py`、`config/task2_config.yaml` 与 `scripts/train_task2_model.py` 等关键模块。

---

## 一、任务目标与依赖

1. **输入**：任务1生成的 `artifacts/task1/source_features.csv`，包含源域分段特征、标签及元数据；
2. **输出**：训练好的源域诊断模型（`source_domain_model.joblib`）、评估指标、可视化、特征重要度报告与多模型对比表；
3. **核心诉求**：
   - 构建线性可解释模型（逻辑回归 + CORAL 对齐），作为后续迁移（任务3）与解释（任务4）的基础；
   - 通过统一 Pipeline 保持数据预处理、对齐与分类器的一致性，方便任务3复用；
   - 生成完备的可解释性与评估产物，以便技术报告撰写与专家复核。

---

## 二、代码结构速览

| 模块 | 关键类/函数 | 职责 |
| --- | --- | --- |
| `src/modeling/task2.py` | `SourceDiagnosisConfig`、`CoralAligner`、`train_source_domain_model` | 定义配置数据类、CORAL 对齐器、训练主流程及各类诊断产出。 |
| | `RealMedianImputer` | 将特征转换为实数并使用中位数填补缺失值，是 Pipeline 的首个步骤。 |
| | `TrainingResult` | 封装训练结果（模型、指标、系数、预测、交叉验证等），供上层调用。 |
| `src/tasks/task2/pipeline.py` | `Task2Config`、`resolve_task2_config` | 将 YAML 配置解析为数据类对象；
| | `run_training` | 调用 `train_source_domain_model`，并负责输出本地化报表、绘图、模型持久化及基线对比。 |
| `scripts/train_task2_model.py` | `run_training` | 命令行入口，支持覆盖特征表和输出目录。 |
| `config/task2_config.yaml` | - | 默认配置，包括数据划分、模型超参、对齐策略、可解释性与基线模型列表。 |

---

## 三、数据准备与特征列解析

1. **列筛选**：
   - `SourceDiagnosisConfig.get_exclude_columns()` 默认剔除 `dataset`、`file_id`、`file_path` 等 9 个元数据列；
   - 若 `feature_columns` 为空，则自动选择剩余的所有数值列；
   - `_drop_near_constant_columns` 会移除只有单一取值的特征，避免无效维度进入模型。
2. **缺失处理**：
   - `RealMedianImputer` 在 `fit` 阶段提取中位数，在 `transform` 时将缺失替换为中位数；
   - 对于可能的复数特征（如某些时频特征），先取实部再填补，并发出警告提示。
3. **标签校验**：
   - 模型预期的类别集合为 `{normal, ball_fault, inner_race_fault, outer_race_fault}`；
   - 若检测到其他标签，将在日志中警告并过滤，确保训练集中只包含标准类别；
   - Stratified split 若失败会自动退化为非分层随机划分，避免因样本稀少导致报错。

---

## 四、训练流水线分解

整体 Pipeline 结构：

```
RealMedianImputer → CoralAligner → StandardScaler → LogisticRegression
```

1. **RealMedianImputer**：确保输入为实数并填补缺失；
2. **CoralAligner**：
   - 在 `fit` 阶段保存源域均值与协方差，计算白化矩阵；
   - 通过 `set_target_statistics` 可在任务3/4 中注入目标域统计量，实现跨域对齐；
   - `get_alignment_summary` 提供协方差迹、条件数、白化后与单位矩阵的 Frobenius 误差，用于评估对齐效果；
3. **StandardScaler**：保证特征均值为 0、方差为 1，提高逻辑回归的数值稳定性；
4. **LogisticRegression**：
   - 默认 `penalty=l2`、`C=1.5`、`class_weight=balanced`、`max_iter=500`，兼容多分类；
   - 系数矩阵 `coef_` 与截距 `intercept_` 在后续解释中被直接引用。

---

## 五、评估与解释性产出

执行 `train_source_domain_model` 后将返回 `TrainingResult`，并经 `run_training` 写出如下成果：

| 产物 | 内容说明 |
| --- | --- |
| `metrics.json` | 记录训练/测试准确率、宏平均 F1、交叉验证均值/标准差以及 CORAL 对齐诊断。 |
| `classification_report.csv` | 本地化后的分类报告（precision/recall/F1/样本数），含中文指标名。 |
| `confusion_matrix.csv` & `confusion_matrix_heatmap.png` | 数值表与可视化图，后者包含原始计数与行归一化对照。 |
| `roc_curves.png` | 支持多分类 ROC，自动绘制 Micro/Macro AUC。 |
| `coefficient_importance.csv` | 每个特征在各类别下的系数、优势比与绝对值，已翻译列名。 |
| `permutation_importance.csv` | 若启用 permutation importance，输出均值/标准差排序结果。 |
| `predictions.csv` | 测试集预测明细（含各类别概率），并提供中文标签列。 |
| `feature_summary.csv` | 训练集中使用特征的描述统计，便于检查异常值。 |
| `features_used.txt` | 保留最终进入模型的特征编码及中文名称，供复现使用。 |
| `source_domain_model.joblib` | 序列化后的 Pipeline，可直接 `joblib.load` 预测。 |
| `model_comparison.csv` | （可选）与随机森林、梯度提升、SVM 等基线模型的指标对比。 |

---

## 六、基线模型对比机制

1. `Task2Config.benchmark_models` 支持在同一数据划分下评估多种算法：
   - 支持类型：`random_forest`、`gradient_boosting`、`svc`、`extra_trees`、`logistic`、`knn`、`mlp`、`gaussian_nb` 等；
   - 每个条目可单独设置 `params`（将在输出表中以 JSON 形式记录）。
2. 评估流程：
   - 复用主模型 Pipeline 的预处理步骤（Imputer + CoralAligner + StandardScaler）；
   - 使用 `train_indices`、`test_indices` 与主模型保持一致的数据划分，确保公平比较；
   - 输出训练/测试准确率及测试宏平均 F1，按测试准确率降序排序。
3. 使用建议：
   - 如果非线性模型表现显著更好，可考虑在任务3引入对应模型并重新设计解释方案；
   - 若仅需查看主模型，请将 `benchmarks` 清空以缩短训练时间。

---

## 七、配置文件详解 (`config/task2_config.yaml`)

| 配置段 | 关键字段 | 说明 |
| --- | --- | --- |
| `features` | `table_path` | 源域特征表路径，允许通过命令行 `--feature-table` 覆盖。 |
| | `label_column` | 目标列，默认 `label`。 |
| | `feature_columns` | 指定使用的特征列表，`null` 表示自动发现。 |
| | `exclude_columns` | 额外排除的列。 |
| `split` | `test_size`、`random_state`、`stratify` | 控制训练/测试划分策略。 |
| `model` | `penalty`、`C`、`solver`、`max_iter`、`class_weight` | 逻辑回归超参。 |
| `alignment` | `enabled`、`epsilon` | 是否启用 CORAL 及其正则项。 |
| `cross_validation` | `enabled`、`folds`、`shuffle` | 控制交叉验证开关与折数。 |
| `interpretability.permutation_importance` | `enabled`、`n_repeats`、`scoring` | permutation importance 设置。 |
| `outputs` | `directory`、`metrics_file` 等 | 控制产物目录与文件名。 |
| `benchmarks` | - | 多模型对比列表。 |

---

## 八、运行方式与日志

- 命令行：
  ```bash
  python scripts/train_task2_model.py \
      --config config/task2_config.yaml \
      --feature-table artifacts/task1/source_features.csv \
      --output-dir artifacts/task2_experiment
  ```
- 日志要点：
  - `INFO`：加载特征、特征列数量、对齐诊断、基线模型训练状态；
  - `WARNING`：标签缺失、分层划分失败、Permutation importance 异常等；
  - `DEBUG`（需手动配置）：可输出更多特征筛选细节。

---

## 九、结果解读建议

1. **指标对比**：重点关注 `metrics.json` 中的训练/测试差距与 `cv_mean_accuracy` 是否稳定；
2. **系数分析**：将 `coefficient_importance.csv` 与任务1的数据字典对照，验证高权重特征的物理含义（例如外圈故障对 BPFO 能量呈正贡献）；
3. **Permutation Importance**：与系数结果交叉验证，识别对分类贡献最大的特征组合；
4. **混淆矩阵/ROC**：观察容易混淆的类别（如内圈 vs 滚动体），为后续伪标签阈值设置提供依据；
5. **特征统计**：若 `feature_summary.csv` 显示特定特征存在极端值，可考虑在任务1阶段重新分段或标准化。

---

## 十、与任务3/4的衔接

- 训练得到的 `SourceDiagnosisConfig` 和 Pipeline 会在任务3中复用，通过 `set_target_statistics` 与伪标签循环实现迁移；
- `coefficient_importance.csv`、`feature_summary.csv` 与 `feature_dictionary.py` 输出形成任务4全局解释的基础；
- `predictions.csv` 的概率列提供伪标签初始筛选的置信度参考。

---

## 十一、常见问题排查

| 问题 | 可能原因 | 对策 |
| --- | --- | --- |
| `ValueError: No labeled samples available` | 输入 CSV 中 `label` 列为空或均缺失 | 检查任务1输出是否包含标签；必要时在配置中指定 `label_column`。 |
| `Stratified split failed` 警告 | 某些类别样本过少 | 可降低 `test_size`、关闭分层或在任务1调整筛选范围。 |
| Permutation importance 抛错 | 测试集样本过少或分类器不支持 `predict_proba` | 增大测试集、关闭 permutation、或改用支持概率输出的基线。 |
| `CoralAligner` 诊断值过大 | 特征高度相关或标准化失败 | 增加 `epsilon`、减少特征数量或启用 PCA 等预处理。 |

---

通过以上说明，读者可以全面理解任务2的建模设计与产物意义，并在需要时定制配置、扩展基线或接入其他解释方法。
