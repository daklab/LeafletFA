from leafletfa.model import LeafletFA

__version__ = "0.1.0"
__all__ = ["LeafletFA", "waypoints", "differential_splicing", "estimate_bayesian_fdr"]


def __getattr__(name):
    if name in ("waypoints", "differential_splicing", "estimate_bayesian_fdr", "utils"):
        import importlib
        return importlib.import_module(f"leafletfa.{name}")
    raise AttributeError(f"module 'leafletfa' has no attribute {name!r}")
