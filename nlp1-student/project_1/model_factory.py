from sklearn.linear_model import LogisticRegression

def model_factory(model: str, **kwargs):
    """Construct a classifier using its model-specific keyword arguments."""
    
    # import lazily to avoid a circular import with smlp_fork.
    
    
    if model in ("linear", "logistic"):
        from project_1.smlp_fork import LogisticRegression
        
        return LogisticRegression(**kwargs)


    if model == "mlp":
        from project_1.smlp_fork import SparseMLP

        return SparseMLP(**kwargs)

    if model == "deep_mlp":
        from project_1.smlp_fork import SparseDeepMLP
        return SparseDeepMLP(**kwargs)

    raise ValueError(f"Unknown model: {model!r}")
