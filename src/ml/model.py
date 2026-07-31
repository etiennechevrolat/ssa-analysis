from torch import nn

### on a en entrée un tenseur (B,F,W) = (batch_size, features, window_size).
## pour un LOCALIZER on veut dim=2 en sortie : manoeuvre EW/NS ? 
## pour un CLASSIFIER on veut un dict en sortie : 
#   une clé node de dim=3 (ID, AD, IK) a priori, une clé classe de dim=4 'NK', 'CK', 'EK', 'HK'


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
    Éléments interessants de leur archi : paraléllisation de l'architecture pour différencier deux types de détection différentes (EW/NS) sur un backbone commun. 
    Leur architecture est développée avec lib Keras, ici adaptée à PyTorch.
    On retient ici un CNN et un LSTM branchés en série. 
    Possibilité de paralléliser, i.e. split en deux channels distinct le cnn ou le lstm via les booléens split_cnn ou split_lstm
    Booléen is_classifier différencie les deux régimes (localizer/classifier) et change la tete de sortie en conséquence.
    """
    def __init__(self, 
                 n_features,
                 window_size,
                 n_node_types = 3,
                 n_classes = 4,
                 is_classifier=False,
                 split_cnn = False,
                 split_lstm = False
                ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.n_node_types = n_node_types
        self.n_classes = n_classes
        self.is_classifier = is_classifier

        ## Les couches de convolution
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

        ## Le pooling se fait toujours sur la dernière dim, avant les couches denses
        self.lstm_pool = nn.MaxPool1d(kernel_size=6, stride=1) 

        self.activation = nn.ReLU()

        ## attention à permuter l'axe temporel pour lstm, attend un tenseur time-first(B,W,F).
        self.lstm_layers = nn.LSTM(48, 64, num_layers=1, batch_first=True) 
        
        self.lstm_dense1 = nn.Linear(64*35, 32) 

        ## Couche dense de localisation 32 -> 2 
        self.lstm_dense2 = nn.Linear(32,2)

        

        ## on modifie la tête pour la classification 
        ## on récupère un tenseur (B, 64*35) après les couches LSTM et reshape
        self.classifier_node_lstm_dense1 = nn.Linear(64*35, 32)
        self.classifier_node_lstm_dense2 = nn.Linear(32, n_node_types)

        self.classifier_classes_lstm_dense1 = nn.Linear(64*35, 32)
        self.classifier_classes_lstm_dense2 = nn.Linear(32, n_classes)

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

        if self.is_classifier : 
            x_node = self.activation(self.classifier_node_lstm_dense1(x)) ## (B, 32)
            x_node = self.classifier_node_lstm_dense2(x_node) ## (B, n_node)

            x_class = self.activation(self.classifier_classes_lstm_dense1(x)) ## (B,32)
            x_class = self.classifier_classes_lstm_dense2(x_class) ## (B, n_classes)

            return {'node' : x_node, 'class' : x_class}
        else : 
            x=self.lstm_dense1(x) ## (B, 32)
            x=self.activation(x)

            x=self.lstm_dense2(x) ## (B,2)

            return x

class small_cnn(nn.Module):
    """
    Un simple CNN de quelques couches pour de la localisation de maneouvres uniquement.
    """
    def __init__(self, 
                 n_features,
                 window_size,
                 split_cnn = False
                ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        

        ## Les couches de convolution
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

        ## Couche dense. en sortie des couches de convolution : (B,F,W) = (256, 48, 40)
        self.dense1 = nn.Linear(48*40, 32)
        self.dense2 = nn.Linear(32, 2)


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

        x= x.reshape(batch_size, 48*40)

        x=self.dense1(x) ## (B,32)
        x=self.activation(x)
        x=self.dense2(x) #(B,2)

        return x


class small_lstm(nn.Module):
    def __init__(self, 
                     n_features,
                     window_size,
                     n_node_types = 3,
                     n_classes = 4,
                     is_classifier=False,
                    ):
            super().__init__()
    
            self.n_features = n_features
            self.window_size = window_size
            self.n_node_types = n_node_types
            self.n_classes = n_classes
            self.is_classifier = is_classifier
    
            ## Le pooling se fait toujours sur la dernière dim, avant les couches denses
            self.lstm_pool = nn.MaxPool1d(kernel_size=6, stride=1) 
    
            self.activation = nn.ReLU()
    
            ## attention à permuter l'axe temporel pour lstm, attend un tenseur time-first(B,W,F).
            self.lstm_layers = nn.LSTM(9, 32, num_layers=1, batch_first=True) 
            
            self.lstm_dense1 = nn.Linear(32*92, 32) 
    
            ## Couche dense de localisation 32 -> 2 
            self.lstm_dense2 = nn.Linear(32,2)
        
    
            ## on modifie la tête pour la classification 
            ## on récupère un tenseur (B, 32*92) après les couches LSTM et reshape
            self.classifier_node_lstm_dense1 = nn.Linear(32*92, 32)
            self.classifier_node_lstm_dense2 = nn.Linear(32, n_node_types)

            self.classifier_classes_lstm_dense1 = nn.Linear(32*92, 32)
            self.classifier_classes_lstm_dense2 = nn.Linear(32, n_classes)
    
    def forward(self, x):
    
        batch_size, features, window_size = x.shape ## (B, F, W) = (256, 9, 97)

        ### Simple couche LSTM, time-first
        x= x.permute(0,2,1) # (Batch, Window, Features) = (B, 97, 9)
        x, (h_n, c_n) = self.lstm_layers(x) # (B, 97, 32)

        # on repermute pour le maxpool 
        
        x = x.permute(0,2,1) ## (B, F, W) = (B, 32, 97)
        x = self.lstm_pool(x) ## (B, 32, 92)  

        x= x.reshape(batch_size, 32*92)

        if self.is_classifier : 
            x_node = self.activation(self.classifier_node_lstm_dense1(x)) ## (B, 32)
            x_node = self.classifier_node_lstm_dense2(x_node) ## (B, n_node_types)

            x_class = self.activation(self.classifier_classes_lstm_dense1(x)) ## (B,32)
            x_class = self.classifier_classes_lstm_dense2(x_class) ## (B, n_classes)

            return {'node' : x_node, 'class' : x_class}
        else : 
            x=self.lstm_dense1(x) ## (B, 32)
            x=self.activation(x)

            x=self.lstm_dense2(x) ## (B,2)

            return x



### On implémente ici une architecture de d'autoencoder via l'approche ViT. 
## Les données massives proviennent de spacetrack x = (Features, Temps). 
# Ce sont des séries temporelles très étalées dans le temps : on peut récupérer les features sur quelques mois, plusieurs années. 
## 1° On divise ces données en patchs non superposés et on projette dans l'espace latent D=16 : 
#       x_1  = (Features, Window), x_2 = (Features, Window), ... , x_p avec la taille de Window à choisir. (Guimareas et al prennent 8)
#  1.5 ° On ajoute des PosEmbedding à ces patchs
#  2° On sample un sous ensemble aléatoire de patch pas trop faible  (40 - 50%).
#  4° On ajoute un token qui représente la données dans sa globalité  : x_class, 
#  5° On fait passer ces données dans un nombre L = 8 de têtes d'attention. 

import torch

class PatchEmbedding(nn.Module):
    def __init__(self, n_features, n_epochs, patch_size=8, embed_dim=16):
        super().__init__()
        assert n_epochs % patch_size == 0
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.patch_size = patch_size
        self.embed_dim =embed_dim

        self.num_patches = n_epochs // patch_size

        self.proj = nn.Conv1d(
            in_channels=n_features,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size, 
            bias=True
        )

    def forward(self, x):
        #x: (B, F, T)
        x = self.proj(x) # (B, embed_dim,  num_patches) le découpage et la projection linéaire sont fait simultanément
        x = x.transpose(1,2) # (B, num_patches, embed_dim)

        return x 

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.PE = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        #x : (B, num_patches, embed_dim)
        batch_size, num_patches, _ = x.shape

        positions = torch.arange(num_patches, device=x.device).unsqueeze(0) # (1, num_patches). unsqueeze add the batch dimension
        # add positional embedding
        x = x+ self.PE(positions) # (B, num_patches, embed_dim)
        return x 
    
import numpy as np 


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout_rate=0.1):
        super().__init__()

        self.x_dim = embed_dim
        self.hidden_dim = embed_dim
        self.n_heads = n_heads

        assert self.hidden_dim % n_heads == 0
        self.head_dim = self.hidden_dim // n_heads

        self.query_matrix = nn.Linear(self.x_dim, self.hidden_dim)
        self.key_matrix = nn.Linear(self.x_dim, self.hidden_dim)
        self.value_matrix = nn.Linear(self.x_dim, self.hidden_dim)

        self.output_proj = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        # x: (B, N, embed_dim)
        batch_size, num_patches, _ = x.shape
  
        Q = self.query_matrix(x) ## (B, N, hidden_dim)
        K = self.key_matrix(x) ## (B, N, hidden_dim)
        V = self.value_matrix(x) ## (B, N, hidden_dim)

        ## principe de l'attention multi-tête:  on projette sur les sous-espaces de l'espace latent 
        Q = Q.view(batch_size, num_patches, self.n_heads, self.head_dim).transpose(1,2) ## (B, num_heads, N, head_dim) 
        K = K.view(batch_size, num_patches, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, num_patches, self.n_heads, self.head_dim).transpose(1,2)

        similarity_matrix = torch.einsum('bdik,bdjk->bdij', Q, K) / self.head_dim**0.5   ## (B, num_heads, N, N)
        attn_weights = torch.softmax(similarity_matrix, dim=-1) ## softmax sur les colonnes pour avoir des poids à passer à V
        attn_weights = self.dropout(attn_weights)
        attn = torch.einsum('bdij,bdjk->bdik', attn_weights, V) ## (B, num_heads, N, head_dim)
        ## merge des têtes
        attn = attn.transpose(1,2).reshape(batch_size, num_patches, self.hidden_dim)
       
        attn = self.output_proj(attn)
        attn = self.dropout(attn)

        return attn

class MLP(nn.Module):
    def __init__(self, in_channels, expansion_factor = 4, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_channels, expansion_factor * in_channels)
        self.dense2 = nn.Linear(expansion_factor * in_channels, in_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x) :
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x=self.dropout(x)
        return x

    
class TransformerBlock(nn.Module):
    def __init__(self, x_len, embed_dim, n_attn_heads, expansion_factor, dropout_rate):
        super().__init__()


        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.MSA = MultiHeadSelfAttention(x_len, embed_dim, n_attn_heads, dropout_rate)
        self.MLP = MLP(embed_dim, expansion_factor, dropout=dropout_rate)


    def forward(self, x):
        # x: (B, N, embed_dim)
        x_id = x 
        x = self.norm1(x)
        attn = self.MSA(x)
        x = x_id  + attn 

        x_id_2 = x 
        x = self.MLP(self.norm2(x))
        x = x +  x_id_2
        return x

class ViTEncoder(nn.Module):
    def __init__(self, 
        n_features,  
        n_epochs, 
        patch_size, 
        embed_dim, 
        n_attn_heads, 
        n_blocks, 
        expansion_factor,
        dropout_rate=0.1
        ):

        super().__init__()

        self.patch_embedding = PatchEmbedding(n_features, n_epochs, patch_size, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        self.pos_embedding = LearnedPositionalEmbedding(embed_dim, self.num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding.PE.weight, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.num_patches, embed_dim, n_attn_heads, expansion_factor, dropout_rate) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        # raw x: (B, n_features, n_epochs)
        x = self.patch_embedding(x) # (B,N,embed_dim)
        x = self.pos_embedding(x) 
        cls =self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat([cls, x], dim=1)
        
        for bloc in self.blocks : 
            x= bloc(x)
        x=self.final_norm(x)
        return x



def build_model(model_cfg, task_cfg, *, n_features, window_size):
    """cfg.model. aiguille vers le bon modèle selon cfg.name"""
    is_classifier = (task_cfg.name == 'classifier')

    
    if model_cfg.name == "naive_baseline":
        return NaiveBaseLine(n_features, window_size)

    heads = {}
    if is_classifier : 
        heads = {"n_node_types" : task_cfg.node_types, "n_classes" : task_cfg.node_classes}
    
    if model_cfg.name == "cnn_lstm1": 
        return cnn_lstm1(
            n_features, 
            window_size, 
            is_classifier=is_classifier, 
            **heads
            )
    
    if model_cfg.name == "small_lstm": 
        return small_lstm(
            n_features, 
            window_size, 
            is_classifier=is_classifier, 
            **heads
            )
    if model_cfg.name == "small_cnn": ## ne fait que du localizer
            return small_cnn(
                n_features, 
                window_size, 
                )
        
    raise KeyError (f"modèle inconnu : {model_cfg.name}")

