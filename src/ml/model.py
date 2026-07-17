from torch import nn

### on a en entrée un tenseur (B,F,W) = (batch_size, features, window_size).
## pour le moment on veut dim=2 en sortie : manoeuvre EW/NS ? 
class NaiveBaseLine(nn.Module):
    def __init__(self, n_features, window_size):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dim_embedding = n_features*window_size
        self.layer1 = nn.Linear(self.dim_embedding, 2) 

    def forward(self, x):
        batch_size, features, window_size = x.shape ## (B, F, W) 

        x = x.reshape(batch_size, self.dim_embedding) ## (B, F*W) = (B, dim_embedding)
        x = self.layer1(x) ## (B, 2)
        return x



def build_model(cfg, *, n_features, window_size):
    """cfg.model. aiguille vers le bon modèle selon cfg.name"""
    if cfg.name == "naive_baseline":
        return NaiveBaseLine(n_features, window_size)
    raise KeyError (f"modèle inconnu : {cfg.name}")

