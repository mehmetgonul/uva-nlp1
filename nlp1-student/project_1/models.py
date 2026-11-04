from torch import nn

class Linear(nn.Module):
    
    def __init__(self, input_dim: int, class_size: int = 3):
        self.linear = nn.Linear(input_dim, class_size)
        
        
    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int, class_size: int = 3):
        
        self.ln_1 = nn.Linear(input_dim, hidden_dim)
        self.ln_2 = nn.Linear(hidden_dim, class_size)
        
        
    def forward(self, x):
        x = self.ln_1(x)
        x = nn.ReLU(x)
        return self.ln_2(x)


class DeepMLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int, class_size: int = 3):
        
        self.ln_1 = nn.Linear(input_dim, hidden_dim)
        self.ln_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_3 = nn.Linear(hidden_dim, class_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.ln_1(x)
        x = nn.GELU(x)
        x = self.dropout(x)
        x = self.ln_2(x)
        x = nn.GELU(x)
        x = self.dropout(x)
        return self.ln_3(x) # logits
    
