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


class cnn_lstm1(nn.Module):
    """Cette classe est inspirée de l'architecture CNN LSTM gagnante du concours détection/classification de manoeuvres sur le dataset SPLID
    Éléments interessants : paraléllisation de l'architecture pour différencier deux types de détection différentes (EW/NS) sur un backbone commun. 
    Architecture développée avec lib Keras adaptée à PyTorch
    """
    def __init__(self, 
                 n_features,
                 window_size
                ):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        
        self.conv1 = nn.Conv1d(in_channels=n_features, 
                               out_channels=64, 
                               kernel_size=7, 
                               stride=1,
                               dilation=1)  ## (B,F,W)  = (256, 9, 97)-> (B,64, W-6)=(256, 64,91)
        
        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=64, 
                               kernel_size=7, 
                               stride=1,
                               dilation=1)  ## (B,64,W-6)=(256, 64, 91) -> (256,64, 85)

        self.conv3 = nn.Conv1d(in_channels=64, 
                               out_channels=48, 
                               kernel_size=7, 
                               stride=2,
                               dilation=1)  ## (B,64,W-12) = (256,64,85) -> (256,48,40)

        self.dense1 = nn.Linear(64,32)
        self.dense2 = nn.Linear(32,2)

    def forward(self, x):
        batch_size, features, window_size = x.shape ## (B, F, W) 
        x= self.conv1(x)
        x= nn.LayerNorm(x)
        x=nn.ReLU(x)

        x= self.conv2(x)
        x= nn.LayerNorm(x)
        x=nn.ReLU(x)

        x= self.conv3(x)
        x= nn.LayerNorm(x)
        x=nn.ReLU(x)
        return x



def build_model(cfg, *, n_features, window_size):
    """cfg.model. aiguille vers le bon modèle selon cfg.name"""
    if cfg.name == "naive_baseline":
        return NaiveBaseLine(n_features, window_size)
    if cfg.name == "cnn_lstm1" : 
        return cnn_lstm1(n_features, window_size)
    raise KeyError (f"modèle inconnu : {cfg.name}")

