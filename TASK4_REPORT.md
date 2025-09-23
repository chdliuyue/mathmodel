# 任务4：迁移诊断可解释性产线全景

本说明聚焦任务4（迁移模型的可解释性分析），结合 `src/tasks/task4/interpretability.py`、`scripts/run_task4_interpretability.py`、`config/task4_config.yaml` 与任务3输出，详细拆解全局/域间/局部解释、聚类对比及产物结构，帮助读者在迁移模型之上建立可信的诊断结论。

---

## 一、框架概览

| 层次 | 目标 | 核心实现 | 输出物 |
| --- | --- | --- | --- |
| 全局解释 | 评估各特征对不同故障类别的总体贡献 | 逻辑回归系数或 SHAP KernelExplainer | `global_feature_effects.csv`、`global_feature_categories.csv`、`global_importance.png` |
| 域间解释 | 量化源/目标域均值差异对 logit 的影响 | `compute_domain_shift_contributions` | `domain_shift_contributions.csv`、`domain_shift.png` |
| 局部解释 | 对单条样本分解特征贡献并可视化 | `explain_samples` + SHAP/LIME/线性贡献 | `local_explanation.csv`、`local_explanation.png` |
| 局部聚类 | 比较多个样本的贡献模式 | `cluster_local_explanations` | `local_cluster_assignments.csv`、`local_cluster_profiles.csv`、`local_cluster_heatmap.png` |

---

## 二、运行入口与依赖关系

1. `scripts/run_task4_interpretability.py` 负责：
   - 解析 YAML，加载任务1特征表；
   - 调用任务3的 `parse_transfer_config` 并执行 `run_transfer_learning` 以获得最新的迁移模型与伪标签；
   - 依次运行全局解释、域间贡献、局部解释、聚类分析，并写出图表/报表；
   - 汇总生成 `interpretability_summary.json`，记录本次解释的关键统计。
2. 因此**任务4会重新执行任务3的迁移流程**：确保配置保持一致，以避免解释与实际模型不符；如需直接复用现有模型，可在代码中扩展加载逻辑。

---

## 三、全局解释模块

### 3.1 数据准备

- `compute_global_feature_effects` 首先通过 `_build_feature_metadata` 调用 `feature_dictionary.py` 获取中文名称与特征类别；
- `build_label_name_map` 将类别编码映射为中文（如 `outer_race_fault` → “外圈故障”）。

### 3.2 线性解释（默认）

- 当分类器提供 `coef_` 时，直接按类别遍历权重：
  - 输出列包括 `类别编码`、`类别`、`特征编码`、`影响值`（系数）、`重要性`（绝对值）、`优势比`；
  - 同时记录截距（标记为 `__intercept__`），在图表中作为“基线贡献”。
- `global_feature_categories.csv` 将特征按类别聚合（时域/频域/包络/故障带/时频等），统计绝对贡献总和，便于宏观对比。

### 3.3 SHAP 解释（可选）

- 当 `method=shap` 或分类器无系数时，使用 Kernel SHAP：
  - `shap.kmeans` 自动压缩背景数据（默认为 200 条），减少计算量；
  - `mean_contribution` 与 `mean_absolute` 分别表示平均贡献值与绝对贡献值；
  - 输出格式与线性解释一致，但“基线贡献”对应 `expected_value`。

### 3.4 可视化

- `plot_global_importance` 从 `global_feature_effects.csv` 中选取绝对贡献 Top-N 特征绘制水平条形图，中文标签来自特征词典；
- 图中可同时展示多个类别（堆叠条形），比较不同故障的贡献差异。

---

## 四、域间贡献分析

1. `compute_domain_shift_contributions` 步骤：
   - 使用迁移结果中的源/目标增强特征，按列计算均值差 `delta`；
   - 将差值与逻辑回归系数相乘，得到每个特征对 logit 的贡献变化 `logit_contribution`；
   - 输出中包含类别编码、中文标签、特征中文名，方便直接写入报告。
2. `plot_domain_shift` 以 Top-N 绝对贡献为横向条形图，区分正负方向，可快速定位导致域偏移的关键特征。
3. 应用建议：
   - 若某些时频特征在目标域产生强烈负向贡献，可能需要在任务3阶段提高一致性阈值或重新对齐；
   - 结合 `pseudo_labels.csv` 检查偏移是否来自特定伪标签样本。

---

## 五、局部解释与聚类

### 5.1 核心流程

1. `explain_samples` 支持三种模式：
   - `method=auto`：线性模型走系数路径，其余模型使用 SHAP；
   - `method=shap`：指定背景样本数 (`shap_background`) 与采样次数 (`shap_nsamples`)；
   - `method=lime`：调用 LIME（需要 scikit-image 等依赖），可通过 `num_samples`、`random_state` 控制稳定性。
2. 输出 `local_explanation.csv`：
   - 包含样本索引、所属数据域（源/目标/伪标签）、目标类别、特征原值/模型输入值、贡献值、贡献绝对值等；
   - `特征中文名` 与 `类别` 均已翻译，便于专家查阅。
3. `plot_local_explanation` 从单个样本的贡献表中选取 Top-N 绘制水平条形图，区分正负贡献。

### 5.2 聚类对比

- 当配置中 `cluster.enabled=true` 且选择了多个样本时：
  - `cluster_local_explanations` 对样本贡献向量执行 KMeans（默认聚类数 = 3），得到 `local_cluster_assignments.csv`；
  - `local_cluster_profiles.csv` 汇总每个聚类的平均贡献 Top-N 特征；
  - `plot_local_cluster_heatmap` 生成热力图，展示不同聚类的贡献模式差异。
- 使用提示：聚类能帮助识别“同类故障但贡献模式不同”的情况，适合在伪标签验证或疑难告警复核中使用。

---

## 六、配置文件关键项 (`config/task4_config.yaml`)

| 配置段 | 字段 | 说明 |
| --- | --- | --- |
| `features` | `source_table`、`target_table` | 引用任务1特征表，支持命令行覆盖。 |
| `modeling` | 同任务3 | 确保与迁移模型一致，以获取可复现的 Pipeline。 |
| `time_frequency`、`pseudo_label` | - | 复用任务3配置，影响重新训练时的时频特征与伪标签行为。 |
| `global_importance` | `method`、`shap_background`、`shap_nsamples`、`top_n` | 控制全局解释方式与绘图 Top-N。 |
| `domain_shift` | `top_n` | 域偏移图展示的特征数量。 |
| `local_explanation` | `dataset`、`indices`、`class`、`method`、`top_n` 以及 SHAP/LIME 子配置 | 指定需要解释的样本、目标类别及方法。 |
| `outputs` | 各类 CSV/PNG/JSON 路径 | 自定义产物存储位置。 |

---

## 七、运行命令与日志

```bash
python scripts/run_task4_interpretability.py \
    --config config/task4_config.yaml \
    --source-features artifacts/task1/source_features.csv \
    --target-features artifacts/task1/target_features.csv \
    --output-dir artifacts/task4_experiment
```

- 日志关注点：
  - `INFO`：迁移流程完成、全局/域间/局部解释写出、聚类结果生成；
  - `WARNING`：SHAP/LIME 依赖缺失、样本索引越界、聚类样本不足等；
  - `DEBUG`（需手动启用）：打印背景样本数量、局部解释特征排序等。

---

## 八、结果解读指南

1. **全局层面**：
   - 对比 `global_feature_categories.csv` 评估各特征族（时域/频域/时频等）的总体重要性；
   - 若发现权重与物理机理不符（如外圈故障的 BPFO 能量为负贡献），需要回溯任务1/3 的特征提取或伪标签质量。
2. **域间层面**：
   - `domain_shift.png` 中的正负条形提示目标域比源域更倾向/更弱的故障模式，可结合现场工况解释；
   - 若某些特征导致明显偏移，可调整任务3的伪标签阈值或在配置中禁用相关特征。
3. **局部层面**：
   - `local_explanation.png` 展示单条告警的主要驱动因子，推荐与任务3的 `多模态时频示例.png` 联合审查；
   - 若多条样本的贡献模式差异大，可使用聚类热力图寻找规律。

---

## 九、常见问题排查

| 现象 | 可能原因 | 建议处理 |
| --- | --- | --- |
| SHAP 运行缓慢或内存占用高 | 背景样本过多、特征维度大 | 降低 `shap_background` 或选择线性解释；必要时先降维。 |
| 局部解释结果为空 | 指定索引不在对应数据域或模型预测失败 | 确认 `dataset`/`indices` 设置正确，检查任务3 产物。 |
| 聚类热力图缺失 | 样本数量不足或聚类关闭 | 增加待解释样本数量，或在配置中关闭 `cluster`。 |
| 域间贡献方向异常 | 目标域数据分布极端或伪标签偏差 | 检查 `pseudo_labels.csv`，必要时重新运行任务3 并调节阈值。 |

---

## 十、与前序任务的协同

- 任务1的数据字典与特征命名直接影响可解释性报表的可读性，保持列名一致至关重要；
- 任务2的 `coefficient_importance.csv` 与任务3的伪标签质量报告是校验解释结果的重要依据；
- 如需在任务4中引入其他模型（如随机森林），可沿用 `run_transfer_learning` 的输出并扩展解释函数以支持特定算法。

---

通过上述说明，使用者可以快速定位任务4各模块的作用、配置参数与产出文件，构建完整的跨域诊断可解释性分析链路。
