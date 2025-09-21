# 任务3：迁移诊断模型设计与实现说明

本说明聚焦“任务3：迁移诊断”，即在任务1/任务2已经构建的特征与源域模型基础上，实现跨域自适应、标签推断与可视化分析。核心成果包括：多模态特征扩展、基于 CORAL 的特征对齐、伪标签自训练、域间差异度量以及端到端的脚本化流程。

## 1. 系统架构概览

```
特征表 (task1) ──▶ 时频增强 ──▶ 源域模型 (task2) ──▶ CORAL 对齐
                                               │
                                               ├─▶ 伪标签循环 (高置信度预测)
                                               │
                                               └─▶ 目标域标定 + 可视化输出
```

所有 orchestrator 代码集中在 `src/tasks/task3/` 下：

| 模块 | 作用 |
| --- | --- |
| `features.py` | 计算 STFT/CWT 统计量，生成 `tf_*` 多模态特征。 |
| `segment_loader.py` | 根据特征表提供的 `file_path/start_sample/end_sample` 动态截取原始时序片段。 |
| `configuration.py` | 解析 YAML 配置，复用任务2的 `SourceDiagnosisConfig`，并扩展时频/伪标签参数。 |
| `transfer.py` | 封装迁移诊断主流程：特征增强、模型训练、CORAL 对齐、伪标签迭代、对齐诊断、预测输出。 |
| `scripts/run_task3_transfer.py` | 命令行入口，串联配置解析、特征读写、可视化导出与模型存档。 |

## 2. 多模态特征设计

1. **时频表征**：
   - **STFT**：提取谱能量、熵、频率/时间矩统计量（均值、标准差、偏度、峭度）以及能量峰所在的时间/频率位置。
   - **CWT**：基于 Ricker 小波获取不同尺度能量分布，计算尺度轴的均值、方差、偏度、峭度及主脊能量。
   - 所有特征统一以 `tf_stft_*`、`tf_cwt_*` 前缀输出，可根据需要在配置中选择性启用/关闭。

2. **兼容性**：
   - 若原始时序缺失或长度不足，系统自动回落为 NaN，并在日志中提示跳过的段数；后续由 `SimpleImputer` 补全，不影响训练。
   - 新增特征同时纳入 `scripts/analyze_features.py` 的统计与翻译词典，任务1/2/3 保持统一特征命名体系。

## 3. 迁移学习策略

1. **源域模型复用**：
   - 直接调用任务2管线（`train_source_domain_model`）训练逻辑回归 + CORAL 对齐模型，免去重复开发。
   - 训练完成后，使用目标域特征均值/协方差更新 CORAL 的 `set_target_statistics`，完成一次性对齐。

2. **伪标签循环**：
   - 在 `confidence_threshold` 设定的置信区间内选择目标域样本，赋予预测标签加入训练集。
   - 每次迭代都会 `clone` 原管线重新拟合，随后重新计算目标域统计量，最多循环 `max_iterations` 次，并限制伪标签总量不超过源域样本的 `max_ratio`。
   - 输出 `pseudo_labels.csv` 和 `pseudo_history.csv` 记录每轮新增样本数、置信度统计等指标。

3. **对齐诊断**：
   - 前/后的特征组合分别保存为 `combined_features_before.csv`、`combined_features_aligned.csv`，并计算 MMD/CORAL 距离（`alignment_before.csv`、`alignment_after.csv`），用于量化域间差异。
   - 自动绘制 t-SNE（before/after），直观展示对齐效果。

## 4. 使用说明

```bash
python scripts/run_task3_transfer.py \
    --config config/task3_config.yaml \
    --source-features artifacts/source_features.csv \
    --target-features artifacts/target_features.csv
```

输出默认保存在 `artifacts/task3/`：

- `target_predictions_initial.csv` / `target_predictions.csv`：目标域段级预测与概率，前者为伪标签前，后者为伪标签后的最终结果。
- `pseudo_labels.csv`、`pseudo_history.csv`：伪标签详情及各轮统计。
- `alignment_before.csv`、`alignment_after.csv`：域对齐指标；配套 `tsne_before.png`、`tsne_after.png`。
- `combined_features_before/aligned.csv`：对齐前后的特征矩阵（含 `dataset` 列），方便后续分析。
- `transfer_model.joblib`：最终迁移模型，可供任务4加载解释。
- `metrics.json`：汇总了源域训练指标、伪标签规模、特征数量等关键信息。

## 5. 结果解读建议

1. **关注对齐效果**：`alignment_after` 中 MMD/CORAL 应显著低于对齐前；若下降不明显，可调节 CORAL epsilon 或增加伪标签迭代。
2. **伪标签质量**：检查 `pseudo_history.csv` 的 `probability_mean/min/max`，若置信度分布过低，需提高阈值或减少迭代次数。
3. **特征贡献**：可以结合任务4的解释性输出，评估时频特征在目标域的贡献度，判断是否需要进一步的多模态扩展（如 GAF、时频图像 CNN 等）。

## 6. 多模态拓展思路

- **原始图像流**：利用已保存的 STFT/CWT 中间矩阵生成热力图（可扩展为第三方 CNN/Transformer 输入）。
- **互信息约束**：在伪标签阶段引入多模态一致性（如时域特征与时频特征的互信息最大化），增强跨域鲁棒性。
- **多源融合**：若有多个目标域，可基于本架构引入多源 CORAL/对抗式自编码器，进一步提升泛化能力。

---

任务3的输出直接馈入任务4的解释性分析，无需额外转换。更多实现细节请参考 `src/tasks/task3/transfer.py` 及 `scripts/run_task3_transfer.py`。
