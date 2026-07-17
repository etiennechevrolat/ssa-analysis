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
                               dilation=1, 
                               bias=False)  ## (B,F,W)  = (256, 9, 97)-> (B,64, W-6)=(256, 64,91)
        
        self.conv2 = nn.Conv1d(in_channels=64, 
                               out_channels=64, 
                               kernel_size=7, 
                               stride=1,
                               dilation=1,
                               bias=False)  ## (B,64,W-6)=(256, 64, 91) -> (256,64, 85)

        self.conv3 = nn.Conv1d(in_channels=64, 
                               out_channels=48, 
                               kernel_size=7, 
                               stride=2,
                               dilation=1,
                               bias=False)  ## (B,64,W-12) = (256,64,85) -> (256,48,40)
        
        ## normalisation sur les features. attend un tenseur channel first (B,F,W).
        self.batchnorm1 = nn.BatchNorm1d(64) 
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(48) 


        self.activation = nn.ReLU()

        ## attention à permuter l'axe temporel pour lstm, attend un tenseur time-first(B,W,F).
        self.lstm_layers = nn.LSTM(48, 64, 1, batch_first=True) 
        
        self.lstm_dense1 = nn.Linear(64*35, 32)
        self.lstm_dense2 = nn.Linear(32,2)

        ## Le pooling se fait toujours sur la dernière dim
        self.lstm_pool = nn.MaxPool1d(kernel_size=6, stride=1) 


    def forward(self, x):
        batch_size, features, window_size = x.shape ## (B, F, W) 

        x= self.conv1(x)
        x= self.batchnorm1(x)
        x= self.activation(x)

        x= self.conv2(x)
        x= self.batchnorm2(x)
        x= self.activation(x)

        x= self.conv3(x)
        x= self.batchnorm3(x)
        x= self.activation(x) ## (Batch, Features, Window) = (B, 48, 40) .

        ### On intègre en série le LSTM, time-first
        x= x.permute(0,2,1) # (Batch, Window, Features) = (B, 40, 48)
        x, (h_n, c_n) = self.lstm_layers(x) 

        # on repermute pour le maxpool 
        
        x = x.permute(0,2,1) ## (B, F, W) = (B, 64, 40)
        x = self.lstm_pool(x) ## (B, 64, 35) 

        x= x.reshape(batch_size, 64*35)

        x=self.lstm_dense1(x) ## (B, 32)
        x=self.activation(x)

        x=self.lstm_dense2(x) ## (B,2)


        return x




def build_model(cfg, *, n_features, window_size):
    """cfg.model. aiguille vers le bon modèle selon cfg.name"""
    if cfg.name == "naive_baseline":
        return NaiveBaseLine(n_features, window_size)
    if cfg.name == "cnn_lstm1" : 
        return cnn_lstm1(n_features, window_size)
    raise KeyError (f"modèle inconnu : {cfg.name}")

