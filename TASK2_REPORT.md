# 任务2：源域故障诊断建模说明

本说明文档用于补充仓库中“任务2：源域故障诊断”相关的整体设计思路、实现细节与使用指南，重点阐述模型的迁移潜力与多层可解释性设计。

## 1. 任务目标回顾

在任务1中，我们已经将原始振动信号转化为结构化特征表（含时域、频域、包络域、故障频带能量等）。任务2的目标是在此基础上：

1. 构建一个能够在源域训练、并为后续跨域迁移奠定基础的诊断模型；
2. 在建模过程中保留并增强可解释性，包括事前（模型结构与输入）、过程（特征对齐与训练日志）与事后（模型输出分析）三层；
3. 对模型效果进行严格量化评估，输出指标与可视化报表以支撑后续分析。

## 2. 数据准备与拆分策略

* 输入数据来源：`artifacts/source_features.csv`，由任务1脚本生成，包含四类典型故障（外圈、内圈、滚动体、正常）及辅助元数据。
* 数据清洗：训练脚本会自动丢弃缺失标签的记录，并剔除数值几乎恒定的特征列（如全部样本RPM相同等），避免无效特征干扰模型。
* 数据划分：默认使用 75% / 25% 的训练-测试划分，优先尝试分层采样以保持每类样本占比一致，若分层失败则自动回退到普通随机划分。
* 交叉验证：训练集内部再通过 `StratifiedKFold`（默认5折）执行稳健性评估，输出均值与方差，确保模型对采样波动不敏感。

## 3. 模型架构与迁移能力设计

训练脚本 `scripts/train_task2_model.py` 基于 scikit-learn Pipeline 构建如下流程：

```
缺失值填补（SimpleImputer, median）
        ↓
CORAL 特征对齐（CoralAligner）
        ↓
标准化（StandardScaler）
        ↓
带类权重的多分类逻辑回归
```

### 3.1 事前可解释性

* **特征筛选**：仅使用任务1生成的物理含义明确的统计与频谱特征，且在建模前自动删除近乎常数的列，保证每个输入维度都具备可解释意义。
* **模型选择**：逻辑回归本身为广泛应用的线性可解释模型，其系数可以直接映射为对各类故障的影响方向与强度。

### 3.2 迁移过程可解释性

* **CORAL 对齐（Correlation Alignment）**：`CoralAligner` 在源域上学习特征均值与协方差，并通过白化+再着色使得输入特征在训练过程中保持零均值和单位协方差。
* **可迁移接口**：`CoralAligner` 提供 `set_target_statistics(mean, covariance)` 方法，可在任务3/4中引入目标域的统计量进行快速对齐，实现轻量迁移学习。
* **对齐诊断**：训练脚本会保存 `source_cov_trace`、`source_cov_condition_number` 与 `whiten_identity_fro_error` 等指标，帮助判断源域特征是否充分被白化。

### 3.3 事后可解释性

* **逻辑回归系数**：输出 `coefficient_importance.csv`，包含每个特征在各故障类别下的系数、绝对值与赔率比，便于分析特征对分类的贡献方向与强度。
* **Permutation Importance**：如配置启用，计算在测试集上的 permutation importance，直观展示各特征对整体性能的边际贡献。
* **预测概率**：`predictions.csv` 保存测试集中每条样本的真实标签、预测标签以及各类别概率，为后续绘制 ROC、可靠性曲线等提供基础。

## 4. 模型评估指标

训练脚本默认输出以下指标：

* **Train/Test Accuracy**：训练集与测试集准确率，评估拟合程度与泛化性能；
* **Macro F1**：各类别 F1 值的算术平均，衡量类别不平衡情形下的整体表现；
* **Classification Report**：细化至每个类别的 Precision、Recall、F1 与 Support；
* **Confusion Matrix**：以 CSV 形式保存，便于直观观察误判模式；
* **Cross-Validation Scores**：若启用交叉验证，输出折均值及标准差，用于衡量模型稳定性。

## 5. 使用指南

1. **确认已完成任务1**：确保 `artifacts/source_features.csv` 已生成；若缺失，可先运行 `python scripts/extract_features.py --config config/dataset_config.yaml`。
2. **查看/调整配置**：`config/task2_config.yaml` 提供了数据划分、模型超参、特征对齐、解释性分析等可调项。常用参数说明如下：
   * `split.test_size`：测试集比例；
   * `model.class_weight`：是否对类别施加平衡权重（推荐 `balanced` 以抵御类别不均衡）；
   * `alignment.enabled`：是否启用 CORAL 对齐，可用于 ablation；
   * `interpretability.permutation_importance`：控制 permutation importance 的重复次数与指标（默认 `f1_macro`）。
3. **运行训练脚本**：

   ```bash
   python scripts/train_task2_model.py --config config/task2_config.yaml
   ```

   可通过 `--feature-table`、`--output-dir` 参数覆盖配置中的默认路径。
4. **查看输出**：训练结果默认写入 `artifacts/task2/`，具体文件说明见 README 表格。
5. **复用模型**：加载 `source_domain_model.joblib` 即可在新数据上执行推理，若需要迁移，可先调用 `CoralAligner.set_target_statistics` 更新目标域统计量后再预测。

## 6. 结果解读建议

* **比较系数与物理机理**：例如 `bpfo_energy` 对外圈故障应呈现正向贡献；若出现反常，可回溯特征提取或数据质量。
* **关注对齐诊断**：`whiten_identity_fro_error` 越接近 0 表明 CORAL 白化越充分；若值较大，说明某些特征仍存在较强相关性，可考虑特征降维或正则化。
* **利用预测概率**：可进一步绘制可靠性曲线、设定告警阈值，或为任务3的伪标签策略提供依据。

## 7. 后续拓展方向

* 在当前线性模型基础上引入非线性方法（如 Gradient Boosting、1D CNN），并保持 CORAL/可解释性模块的可复用性；
* 结合目标域少量标注样本，验证 `CoralAligner` 的快速自适应能力，并探索多源统计量融合策略；
* 将 permutation importance 与任务1中的特征重要性可视化结果结合，形成端到端的特征可信度评估框架。

### 2025-09-21 架构升级小结

- **模块分层**：建模逻辑迁移至 `src/tasks/task2/pipeline.py`，配合 `scripts/train_task2_model.py`，构成“配置解析 + 训练执行 + 产出写入”的标准流程，上游任务（task1）输出的 CSV 可无缝复用。
- **迁移接口**：`run_transfer_learning`（task3）直接依赖 `SourceDiagnosisConfig`，无须重新实现模型训练；源域模型训练完毕即可接入时频增强特征与伪标签策略。
- **模型对比**：在同一训练/测试划分下自动评估随机森林、梯度提升与支持向量机等基线模型，输出《model_comparison.csv》便于撰写对比分析。
- **数据衔接**：新增的时间频率特征、故障频率校验表及特征翻译词典在任务2阶段已全部可用，下游任务若不需要可在配置中关闭；整条流水线保持“向后兼容，向前丰富”的设计目标。

---

如需更深入了解实现细节或二次开发，可直接查阅 `src/modeling/task2.py` 与 `scripts/train_task2_model.py` 源码。
