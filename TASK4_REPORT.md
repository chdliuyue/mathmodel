# 任务4：迁移诊断可解释性分析说明

本说明针对“任务4：迁移诊断可解释性”，展示如何在任务3的迁移模型基础上，从全局、域间、样本级三个层面对模型决策进行解释，并生成图表/数据报告，提升诊断人员的透明度与信任度。

## 1. 可解释性框架

| 层次 | 工具 | 输出 | 目的 |
| --- | --- | --- | --- |
| 全局 | 逻辑回归系数 + 特征翻译词典 | `global_feature_effects.csv`、`global_feature_categories.csv`、`global_importance.png` | 识别对各故障类别最关键的特征，理解其物理含义。 |
| 域间 | 特征均值差值 × 系数 | `domain_shift_contributions.csv`、`domain_shift.png` | 定量评估源/目标域分布差异对模型 logit 的影响方向与幅度。 |
| 局部 | 特征贡献分解 | `local_explanation.csv`、`local_explanation.png` | 对单条样本给出“贡献度排序”，辅助专家验证模型判断。 |

底层实现位于 `src/tasks/task4/interpretability.py`，主要函数包括：

- `compute_global_feature_effects`：统计每个类别/特征的系数、绝对系数与赔率比，并按特征类别聚合。
- `compute_domain_shift_contributions`：比较源/目标域特征均值差异，乘以系数得到 logit 贡献；输出中文类别列并在绘图时按“特征×类别”分组，便于定位造成域偏移的关键因素。
- `explain_instance`：提取单条样本经 pipeline 变换后的特征向量，与系数相乘得到 logit 贡献，同时保留原始取值与目标类别中文名称，便于理解幅度。
- 三个 `plot_*` 方法负责绘制水平条形图，默认自动控制纵轴密度，保证在多个特征时仍易于阅读。

## 2. 运行方式

```bash
python scripts/run_task4_interpretability.py \
    --config config/task4_config.yaml \
    --source-features artifacts/source_features.csv \
    --target-features artifacts/target_features.csv
```

配置与任务3保持一致，若有额外需求（如更换局部解释的样本索引/类别），可在 `local_explanation` 节下调整。输出目录默认为 `artifacts/task4/`：

- `global_feature_effects.csv`：按类别列出所有特征的系数、绝对值、赔率比，并附上中文释义与特征类别。
- `global_feature_categories.csv`：按类别汇总绝对系数和，便于宏观判断“时域/频域/包络/时频”等模块的重要性。
- `domain_shift_contributions.csv`：每个类别下的 `logit_contribution` 与 `delta`（均值差），配合 `domain_shift.png` 的分组水平条形图快速识别正负影响。
- `local_explanation.csv`：指定样本的特征贡献明细，包括原始值、模型输入值、系数、logit 贡献及目标类别中文名称，辅助专家复核。
- 三张 PNG 图对应该三张表格，便于直接放入报告或演示文稿。
- `interpretability_summary.json` 汇总了特征数量、伪标签规模、局部解释所选样本等关键信息。

## 3. 成果核查与实践经验

- **全局解释审阅**：打开 `global_feature_effects.csv` 后，可按“重要性”列排序，确认关键特征与任务1/2的数据字典保持一致。配套的 `global_importance.png` 采用中文标注，可直接插入汇报材料。
- **域偏移排查**：`domain_shift_contributions.csv` 的 `Logit贡献` 列越大，说明该特征对域偏移影响越强。结合 `domain_shift.png` 的正负条形图，可以迅速定位需要重新对齐或归一化的特征。
- **局部解释复核**：`local_explanation.csv` 中的 `贡献值` 列与 `local_explanation.png` 的条形图互为验证。建议将同一故障的多个样本加入 `indices` 配置，生成 `local_cluster_heatmap.png`，观察贡献模式是否稳定。
- **指标总结**：`interpretability_summary.json` 提供本次解释任务的统计信息（特征数量、伪标签规模、聚类数量等），方便在实验记录表中登记，形成可追溯的实验链路。

## 4. 分析建议

1. **全局层面**：
   - 对比 `global_feature_effects.csv` 与任务1/2的特征翻译词典，验证高权重特征是否符合轴承机理（如 `fault_bpfo_band_energy` 对外圈故障应为正贡献）。
   - 若发现“反常”权重，可回溯对应特征的提取逻辑或伪标签质量。

2. **域间层面**：
   - 重点关注 `logit_contribution` 绝对值大的特征，若某一特征在目标域显著偏移且导致 logit 增大，可考虑在任务3增加该特征的对齐力度或做归一化。
   - `delta` 的符号揭示了目标域均值与源域的偏离方向，结合伪标签可判断该偏移是否合理。

3. **局部层面**：
   - 通过 `local_explanation.csv` 可以快速锁定某条告警的关键特征，若贡献主要来自时频特征，说明多模态增强发挥作用。
   - 可多选几个样本（修改配置中的 `index` 和 `class`）比较不同故障的贡献模式，以形成经验库。

## 5. 与任务3的衔接

- `scripts/run_task4_interpretability.py` 内部会复用任务3相同的配置解析逻辑并重新运行 `run_transfer_learning`，保证解释结果与最新模型保持一致。
- 若已执行任务3并生成 `transfer_model.joblib` 等输出，可直接将其作为先验，或在任务4中调整配置再次训练，以便开展“对照试验”（例如关闭伪标签、仅保留时域特征等）。

## 6. 后续拓展方向

- **更细粒度的局部解释**：可在 `explain_instance` 基础上，引入 SHAP/LIME 等方法，对非线性模型或深度模型进行扩展。
- **跨样本对比**：对多条样本的贡献结果进行聚类或可视化，挖掘不同故障类型在目标域的共性差异。
- **人机交互面板**：利用导出的 CSV/PNG，可进一步开发交互式看板，实现“点击样本 → 展示贡献度 → 调整阈值”的工作流。

---

任务4提供了贯通全局-局部、多模态-时域的解释性框架，为后续撰写建模报告、与领域专家沟通提供了坚实的数据与可视化支撑。
