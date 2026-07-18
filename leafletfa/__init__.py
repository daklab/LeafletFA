from leafletfa.model import LeafletFA

__version__ = "0.1.0"
__all__ = [
    "LeafletFA",
    "waypoints",
    "differential_splicing",          # factor-level DS (program characterisation)
    "differential_splicing_groups",   # group-vs-group DS (the primary marker analysis)
    "estimate_bayesian_fdr",
    "utils",
]

_LAZY = (
    "waypoints",
    "differential_splicing",
    "differential_splicing_groups",
    "estimate_bayesian_fdr",
    "utils",
)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return importlib.import_module(f"leafletfa.{name}")
    raise AttributeError(f"module 'leafletfa' has no attribute {name!r}")
