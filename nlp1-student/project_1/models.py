from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def model_factory(model: str, hidden_dim: int, act_fn):
    
    if model == "linear":
        return linear_model()
    
    if model == "mlp":
        return mlp(hidden_dim, act_fn)
    
    if model == "deep_mlp":
        return deep_mlp(hidden_dim, act_fn)

def linear_model(seed=42):
    return LogisticRegression(
        solver="saga",
        random_state=seed
    )
    
def mlp(hidden_dim: int = 128, act_fn: str = "relu", seed=42):
    return MLPClassifier(
        hidden_layer_sizes=(hidden_dim, ),
        activation=act_fn,
        solver="adam",
        learning_rate=1e-4,
        alpha=1e-4,
        iter=10,
        random_state=seed
    )
    
def deep_mlp(hidden_dim: int = 128, act_fn: str = "relu", seed=42):
        return MLPClassifier(
        hidden_layer_sizes=(hidden_dim, hidden_dim),
        activation=act_fn,
        solver="adam",
        learning_rate=1e-4,
        alpha=1e-4,
        iter=10,
        random_state=seed,
    )