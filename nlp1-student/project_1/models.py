from torch import nn


def get_act_fn(act_fn: str):

    if act_fn == "relu":
        return nn.ReLU()

    if act_fn == "gelu":
        return nn.GELU(approximate='tanh')

    raise ValueError(f"Unsupported activation function: {act_fn!r}. Choose 'relu' or 'gelu'.")

class Linear(nn.Module):

    def __init__(self, input_dim: int, class_size: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, class_size)


    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim:int, class_size: int = 3, act_fn: str = "relu"):
        super().__init__()

        self.ln_1 = nn.Linear(input_dim, hidden_dim)
        self.ln_2 = nn.Linear(hidden_dim, class_size)

        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.act_fn(x)
        return self.ln_2(x)


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim:int, class_size: int = 3, act_fn: str = "relu"):
        super().__init__()

        self.ln_1 = nn.Linear(input_dim, hidden_dim)
        self.ln_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_3 = nn.Linear(hidden_dim, class_size)

        self.act_fn = get_act_fn(act_fn)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.ln_2(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return self.ln_3(x) # logits

