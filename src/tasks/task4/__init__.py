"""Public interface for task 4 interpretability helpers."""
from .interpretability import (
    GlobalInterpretabilityResult,
    cluster_local_explanations,
    compute_domain_shift_contributions,
    compute_global_feature_effects,
    explain_instance,
    explain_samples,
    plot_domain_shift,
    plot_global_importance,
    plot_local_cluster_heatmap,
    plot_local_explanation,
)

__all__ = [
    "GlobalInterpretabilityResult",
    "cluster_local_explanations",
    "compute_domain_shift_contributions",
    "compute_global_feature_effects",
    "explain_instance",
    "explain_samples",
    "plot_domain_shift",
    "plot_global_importance",
    "plot_local_cluster_heatmap",
    "plot_local_explanation",
]
