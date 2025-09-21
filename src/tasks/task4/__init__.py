"""Public interface for task 4 interpretability helpers."""
from .interpretability import (
    GlobalInterpretabilityResult,
    compute_domain_shift_contributions,
    compute_global_feature_effects,
    explain_instance,
    plot_domain_shift,
    plot_global_importance,
    plot_local_explanation,
)

__all__ = [
    "GlobalInterpretabilityResult",
    "compute_domain_shift_contributions",
    "compute_global_feature_effects",
    "explain_instance",
    "plot_domain_shift",
    "plot_global_importance",
    "plot_local_explanation",
]
