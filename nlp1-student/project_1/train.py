from torch import Data

def train(model: str, data: ):
    
    if model not in {"linear", "mlp", "deepmlp"}:
        return 