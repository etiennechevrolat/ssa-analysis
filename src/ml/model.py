import torch
from torch import nn


def conv1d_out_len(w, kernel_size, stride=1, dilation=1):
    """Longueur de sortie d'une Conv1d/MaxPool1d (padding=0)."""
    return (w - dilation * (kernel_size - 1) - 1) // stride + 1

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

        ## longueur temporelle après les 3 convs puis le maxpool (ex : W=97 -> 40 -> 35)
        w = window_size
        for k, s in ((7, 1), (7, 1), (7, 2), (6, 1)):
            w = conv1d_out_len(w, k, s)
        self.flat_dim = 64 * w

        self.lstm_dense1 = nn.Linear(self.flat_dim, 32)

        ## Couche dense de localisation 32 -> 2
        self.lstm_dense2 = nn.Linear(32,2)

        ## on modifie la tête pour la classification
        ## on récupère un tenseur (B, flat_dim) après les couches LSTM et reshape
        self.classifier_node_lstm_dense1 = nn.Linear(self.flat_dim, 32)
        self.classifier_node_lstm_dense2 = nn.Linear(32, n_node_types)

        self.classifier_classes_lstm_dense1 = nn.Linear(self.flat_dim, 32)
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

        x= x.reshape(batch_size, self.flat_dim)

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
        w = window_size
        for k, s in ((7, 1), (7, 1), (7, 2)):
            w = conv1d_out_len(w, k, s)
        self.flat_dim = 48 * w
        self.dense1 = nn.Linear(self.flat_dim, 32)
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

        x= x.reshape(batch_size, self.flat_dim)

        x=self.dense1(x) ## (B,32)
        x=self.activation(x)
        x=self.dense2(x) #(B,2)

        return x


class cnn_split(nn.Module):
    """CNN pur à backbone parallélisé par direction, d'après la solution gagnante du concours SPLID.

    Le modèle retenu par le gagnant (localizer_sweeper.py : lstm_layers=[], split_cnn=True)
    n'utilise aucune couche récurrente : deux piles de convolutions **indépendantes**
    traitent la même fenêtre d'entrée, l'une dédiée aux noeuds EW, l'autre aux noeuds NS.
    Les manoeuvres in-plane (signature sur a) et out-of-plane (signature sur i) ayant des
    signatures très différentes, un backbone commun force un compromis qui pénalise la
    direction la moins représentée — ici NS.

    Chaque bloc suit l'ordre du gagnant : Conv1d -> ReLU -> BatchNorm -> Dropout
    (la normalisation est appliquée APRÈS l'activation).
    La pile est ici plus profonde que celle du gagnant (5 blocs contre 3), avec de la
    dilatation pour élargir le champ réceptif sans multiplier les paramètres.

    conv_layers : liste de (canaux, noyau, stride, dilatation, maxpool)
    split_dense : si False (défaut, comme le gagnant), les deux branches sont concaténées
                  et partagent la tête dense, chaque direction gardant sa propre sortie.
    """

    def __init__(self,
                 n_features,
                 window_size,
                 conv_layers=((64, 7, 1, 1, 1),
                              (64, 7, 1, 1, 1),
                              (96, 7, 1, 2, 2),
                              (96, 7, 1, 2, 2),
                              (128, 5, 1, 1, 1)),
                 dense_layers=(64, 32),
                 dropout=0.05,
                 split_dense=False,
                ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.split_dense = split_dense

        def build_branch():
            layers, in_channels, w = [], n_features, window_size
            for out_channels, kernel, stride, dilation, maxpool in conv_layers:
                layers += [
                    nn.Conv1d(in_channels, out_channels, kernel,
                              stride=stride, dilation=dilation, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout(dropout),
                ]
                w = conv1d_out_len(w, kernel, stride, dilation)
                if maxpool > 1:
                    layers.append(nn.MaxPool1d(maxpool))
                    w = conv1d_out_len(w, maxpool, maxpool)
                in_channels = out_channels
            if w < 1:
                raise ValueError(
                    f"fenêtre trop courte ({window_size}) pour cette pile de convolutions")
            return nn.Sequential(*layers), in_channels * w

        ## un backbone par direction, sans aucun poids partagé
        self.branch_ew, flat_dim = build_branch()
        self.branch_ns, _ = build_branch()
        self.flat_dim = flat_dim

        def build_head(in_dim):
            layers = []
            for units in dense_layers:
                layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = units
            return nn.Sequential(*layers), in_dim

        if split_dense:
            self.head_ew, out_dim = build_head(flat_dim)
            self.head_ns, _ = build_head(flat_dim)
        else:
            ## têtes denses partagées sur la concaténation des deux branches
            self.head, out_dim = build_head(2 * flat_dim)

        ## une sortie scalaire par direction
        self.out_ew = nn.Linear(out_dim, 1)
        self.out_ns = nn.Linear(out_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]  ## (B, F, W)

        z_ew = self.branch_ew(x).reshape(batch_size, self.flat_dim)
        z_ns = self.branch_ns(x).reshape(batch_size, self.flat_dim)

        if self.split_dense:
            z_ew, z_ns = self.head_ew(z_ew), self.head_ns(z_ns)
        else:
            z_ew = z_ns = self.head(torch.cat([z_ew, z_ns], dim=1))

        return torch.cat([self.out_ew(z_ew), self.out_ns(z_ns)], dim=1)  ## (B, 2)


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
            self.lstm_layers = nn.LSTM(n_features, 32, num_layers=1, batch_first=True)

            self.flat_dim = 32 * conv1d_out_len(window_size, 6)

            self.lstm_dense1 = nn.Linear(self.flat_dim, 32)

            ## Couche dense de localisation 32 -> 2
            self.lstm_dense2 = nn.Linear(32,2)

            ## on modifie la tête pour la classification
            ## on récupère un tenseur (B, flat_dim) après les couches LSTM et reshape
            self.classifier_node_lstm_dense1 = nn.Linear(self.flat_dim, 32)
            self.classifier_node_lstm_dense2 = nn.Linear(32, n_node_types)

            self.classifier_classes_lstm_dense1 = nn.Linear(self.flat_dim, 32)
            self.classifier_classes_lstm_dense2 = nn.Linear(32, n_classes)
    
    def forward(self, x):
    
        batch_size, features, window_size = x.shape ## (B, F, W) = (256, 9, 97)

        ### Simple couche LSTM, time-first
        x= x.permute(0,2,1) # (Batch, Window, Features) = (B, 97, 9)
        x, (h_n, c_n) = self.lstm_layers(x) # (B, 97, 32)

        # on repermute pour le maxpool 
        
        x = x.permute(0,2,1) ## (B, F, W) = (B, 32, 97)
        x = self.lstm_pool(x) ## (B, 32, 92)

        x= x.reshape(batch_size, self.flat_dim)

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

class PatchEmbedding(nn.Module):
    def __init__(self, n_features, window_size, patch_size=8, embed_dim=16):
        super().__init__()
        assert window_size % patch_size == 0
        self.n_features = n_features
        self.window_size = window_size
        self.patch_size = patch_size
        self.embed_dim =embed_dim

        self.num_patches = window_size // patch_size

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
    def __init__(self, embed_dim, n_attn_heads, expansion_factor, dropout_rate):
        super().__init__()


        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.MSA = MultiHeadSelfAttention(embed_dim, n_attn_heads, dropout_rate)
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

class VanillaViT(nn.Module):
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
            [TransformerBlock(embed_dim, n_attn_heads, expansion_factor, dropout_rate) for _ in range(n_blocks)]
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

## On utilise notre ViT précédent pour construire des modèles avec de l'attention

class small_vit(nn.Module):
    """
    Un premier modèle avec à peu près 200k params
    """
    def __init__(self, n_features, window_size,
        patch_size = 8,
        embed_dim = 64, 
        n_attn_heads=4,
        n_blocks = 4,
        expansion_factor = 4,
        dropout_rate=0.1
        ):
        super().__init__()

        self.encoder = VanillaViT(n_features, window_size, patch_size,embed_dim, n_attn_heads, n_blocks, expansion_factor, dropout_rate)

        self.dense1 = nn.Linear(embed_dim, 32)
        self.dense2 = nn.Linear(32, 2)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, n_features, n_epochs) en entrée
        out = self.encoder.forward(x) # (B, N + 1, embed_dim)
        z = out[:,0] # (B, embed_dim)
        z = self.dense2(self.activation(self.dense1(z))) # (B, 2)
        return z 


class InstanceNormMixin:
    """RevIN par fenetre : buffers, configuration et convention sur sigma.

    Partage entre le pretrain (TimeSeriesMAE) et le finetuning (ManeuverFineTune) : l'encodeur
    gele n'a vu que des fenetres centrees reduites par ces statistiques, lui en envoyer d'autres
    revient a le faire travailler hors distribution. Une seule definition, donc, pour que les
    deux ne puissent pas diverger.

    Les buffers restent au premier niveau du state_dict (inorm_mask, sigma_floor) : c'est le nom
    que filtrent deja inference.py et les checkpoints anterieurs a leur introduction.
    """

    def _register_instance_norm(self, n_features):
        self.register_buffer('inorm_mask', torch.zeros(n_features)) ## 1.0 sur les features normalisées par fenetre
        self.register_buffer('sigma_floor', torch.zeros(n_features))

    def configure_instance_norm(self, feature_cols, cols=(), sigma_floor=0.0):
        """Active le RevIN par fenetre sur les canaux `cols`.

        sigma_floor est exprime dans les unites du scaler amont (celui du dataset, global
        ou par objet) : c'est lui qui empeche de ramener a variance 1 une fenetre dont la
        variation est au niveau du bruit. Scalaire, ou dict {canal: valeur}.
        """
        feature_cols = list(feature_cols)
        mask = torch.zeros(len(feature_cols))
        floor = torch.zeros(len(feature_cols))
        for c in cols:
            if c not in feature_cols:
                raise ValueError(f"canal inconnu pour le RevIN par fenetre : {c!r} "
                                 f"(disponibles : {list(feature_cols)})")
            i = feature_cols.index(c)
            mask[i] = 1.0
            floor[i] = float(sigma_floor[c] if hasattr(sigma_floor, 'get') else sigma_floor)
        self.inorm_mask.copy_(mask.to(self.inorm_mask.device))
        self.sigma_floor.copy_(floor.to(self.sigma_floor.device))
        return self

    def _mask_stats(self, mu, sigma):
        """Plancher sur sigma, puis neutralisation des canaux hors inorm_mask (mu=0, sigma=1).

        Les canaux non selectionnes ressortent inchanges, et le chemin de code reste unique
        que le RevIN par fenetre soit actif ou non.
        """
        sigma = torch.clamp(torch.maximum(sigma, self.sigma_floor[None, :]), min=1e-6)
        m = self.inorm_mask[None, :]
        return mu * m, sigma * m + (1.0 - m)


class TimeSeriesMAE(InstanceNormMixin, nn.Module):
    """
    Architecture de masked-auto-encoder pour times series, inspiré de Guimaraes, adapté de VideoMAEv2
    """
    def __init__(self, 
            n_features,  
            window_size, 
            patch_size, 
            encoder_embed_dim, 
            encoder_n_attn_heads, 
            encoder_n_blocks, 
            decoder_embed_dim,
            decoder_n_attn_heads, 
            decoder_n_blocks,
            expansion_factor,
            masking_ratio = 0.5,
            dropout_rate=0.1
            ):
            
            super().__init__()

            self.patch_embedding = PatchEmbedding(n_features, window_size, patch_size, encoder_embed_dim)
            self.num_patches = self.patch_embedding.num_patches
            self.encoder_pos_embedding = LearnedPositionalEmbedding(encoder_embed_dim, self.num_patches)
            self.decoder_pos_embedding = LearnedPositionalEmbedding(decoder_embed_dim, self.num_patches+1)

            self.cls_token = nn.Parameter(torch.zeros(1,1, encoder_embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.encoder_pos_embedding.PE.weight, std=0.02)
            self.mask_token = nn.Parameter(torch.zeros(1,1, decoder_embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            nn.init.trunc_normal_(self.decoder_pos_embedding.PE.weight, std=0.02)

            self.encoder_blocks = nn.ModuleList(
                [TransformerBlock(encoder_embed_dim, encoder_n_attn_heads, expansion_factor, dropout_rate) for _ in range(encoder_n_blocks)]
            )

            self.decoder_blocks = nn.ModuleList(
                            [TransformerBlock(decoder_embed_dim, decoder_n_attn_heads, expansion_factor, dropout_rate) for _ in range(decoder_n_blocks)]
                        )

            self.masking_ratio = masking_ratio 

            self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
            self.encoder_to_decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            
            self.pred_space_proj = nn.Linear(decoder_embed_dim, n_features * patch_size)
            self._register_instance_norm(n_features)

    def _instance_stats(self, x_id, ids_keep) :
        """ mean, std for each (window, feature) sur les patchs visibles seulement

        Les statistiques sont calculees sur les patchs VISIBLES et non sur toute la fenetre :
        sinon on donnerait au decodeur la moyenne et l'echelle des patchs masques, qui
        retrouverait leur niveau sans rien avoir appris.

        Les canaux hors inorm_mask ressortent avec mu=0 et sigma=1 : ils sont inchanges, et
        le chemin de code reste unique que le RevIN par fenetre soit actif ou non.
        """
        B, n_features ,  _ = x_id.shape
        patch_size = self.patch_embedding.patch_size

        patches = x_id.reshape(B, n_features, self.num_patches, patch_size)
        idx = ids_keep[:, None, :, None].expand(B, n_features, ids_keep.shape[1], patch_size)
        visible = torch.gather(patches, 2, idx)                    # (B, F, N_keep, P)

        mu = visible.mean(dim=(2, 3))                              # (B, F)
        sigma = x_id.std(dim=2, unbiased=False)
        return self._mask_stats(mu, sigma)

    def forward(self, x):
        # raw x: (B, n_features, window_size)
        x_id = x
        B, n_features, _ = x.shape
        N = self.num_patches

        ## Masking. Tire AVANT le patch embedding : la normalisation par fenetre ne doit voir
        ## que les patchs visibles, donc elle a besoin de ids_keep. Le tirage ne depend que de
        ## (B, N), pas du contenu de x, donc le deplacer ne change rien au masquage.
        n_mask = round(N * self.masking_ratio)
        n_keep = N - n_mask
        ## rand crée un tenseur aléatoire de shape (B,N), et argsort donne les indices pour trier la liste :  i.e. une permu aléatoire de (1,..,N)*
        ### le deuxième argsort donne la permutation inverse, utile pour reconstruction 
        rand = torch.rand(B,N, device=x.device) #(B,12)
        ids_shuffle = torch.argsort(rand, dim=1) #(B, 12) 
        ids_restore = torch.argsort(ids_shuffle, dim=1) #(B,12)
 
        ids_keep = ids_shuffle[:, : n_keep] # (B, N_keep)
        ids_mask = ids_shuffle[:, n_keep:]

        ## RevIN par fenetre : chaque (fenetre, canal) selectionne par inorm_mask est centre
        ## reduit par ses propres statistiques. L'entree ET la cible en heritent (target_all
        ## est construit depuis x_id), donc la loss devient sans echelle : une fenetre calme
        ## et un objet en decroissance pesent le meme poids.
        mu, sigma = self._instance_stats(x_id, ids_keep)
        x_id = (x_id - mu[..., None]) / sigma[..., None]
        ## conserves pour l'evaluation : ils permettent de repasser en unites physiques.
        self.last_mu, self.last_sigma = mu.detach(), sigma.detach()

        x = self.patch_embedding(x_id) # (B,N,embed_dim)
        x = self.encoder_pos_embedding(x)
        embed_dim = x.shape[-1]

        ids_keep = ids_keep.unsqueeze(-1).expand(-1,-1, embed_dim) #(B, N_keep, embed_dim)
        x_vis = torch.gather(x, dim=1, index=ids_keep) # (B, N_keep, embed_dim)

        

        # Encoder ne voit que les patchs visibles
        cls =self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x_vis], dim=1) # (B, N_keep + 1, embed_dim)
        for bloc in self.encoder_blocks : 
            h= bloc(h)
        h=self.encoder_norm(h)

        # Projection enc -> dec et séquence pour le décodeur
        y = self.encoder_to_decoder_embedding(h) #(B, N_keep+1, decoder_embed_dim)

        _, _, decoder_embed_dim = y.shape
        mask_tokens = self.mask_token.expand(B, n_mask, -1) # (B, N_masked, decoder_dim), ordre mélangé
        # on concatène les tokens visibles (sans le cls token) avec les tokens de mask
        y_ = torch.cat([y[:,1:], mask_tokens], dim=1) # (B, N, decoder_dim)
        # on reordonne selon la séquence initiale 
        y_ = torch.gather(y_, 1, ids_restore.unsqueeze(-1).expand(-1,-1,decoder_embed_dim))


        ## Decodeur on reconstitue la séquence puis on ajoute le PE
        y = torch.cat([y[:,:1],y_], dim=1) 
        y = self.decoder_pos_embedding(y)
        

        for block in self.decoder_blocks:
            y = block(y)
        y =self.decoder_norm(y)

        # proejction vers l'espace des patchs (sans le cls)
        pred_all = self.pred_space_proj(y[:, 1:])  #(B,N,F*P)

        #Construction cible
        target_all = (
            x_id.reshape(B,n_features, self.num_patches, -1)
                .permute(0,2,1,3)
                .reshape(B,self.num_patches, -1)
                )

        ## on ne regarde que la loss sur les patchs maskés
        ids_mask = ids_mask.unsqueeze(-1).expand(-1,-1, n_features * self.patch_embedding.patch_size)
        pred = torch.gather(pred_all, 1, ids_mask) # (B, N_mask, F*P)
        target = torch.gather(target_all, 1, ids_mask)
        return pred, target


    ## correspondance MAE -> VanillaViT
    ENCODER_KEY_MAP = {
        "patch_embedding." : "patch_embedding.",
        "encoder_pos_embedding." : "pos_embedding.",
        "encoder_blocks." : "blocks.",
        "encoder_norm." : "final_norm."
        }
    
    def encoder_state_dict(self) : 
        """Poids de l'encodeur seul renommé au format VanillaViT"""
        out={}
        for k,v in self.state_dict().items():
            if k =="cls_token":
                out[k] = v
                continue
            for src, dst in self.ENCODER_KEY_MAP.items():
                if k.startswith(src):
                    out[dst + k[len(src):]] = v
                    break
        return out 


class ManeuverFineTune(InstanceNormMixin, nn.Module):
    """
    On recopie l'architecture du MAE (v1 a v3) pour finetune sur des manoeuvres :
    """
    def __init__(self, n_features, window_size, patch_size, embed_dim, n_attn_heads, n_blocks, expansion_factor, dropout_rate, freeze_encoder, n_outputs=4):
        super().__init__()
        self._register_instance_norm(n_features)
        self.encoder = VanillaViT(n_features, 
                                  window_size, 
                                  patch_size,
                                  embed_dim, 
                                  n_attn_heads, 
                                  n_blocks, 
                                  expansion_factor, 
                                  dropout_rate)
        
        ## Le CLS résume TOUTE la fenêtre (192 TLE, soit 50 à 190 jours selon la cadence de
        ## l'objet) alors que la cible porte sur son point central : il n'a aucune spécificité
        ## positionnelle. On lui concatène le token du patch central, qui contient le centre
        ## de la fenêtre, d'où l'entrée de dense1 en 2*embed_dim.
        self.center_patch = 1 + self.encoder.num_patches // 2 ## +1 : le CLS occupe l'indice 0
        self.dense1 = nn.Linear(2 * embed_dim, 32)
        self.dense2 = nn.Linear(32, n_outputs)
        self.activation = nn.GELU()
        self.freeze_encoder = freeze_encoder

    def _window_stats(self, x):
        """mu, sigma par (fenetre, canal), sur la fenetre entiere.

        Difference assumee avec le pretrain : le MAE calcule mu sur les seuls patchs visibles
        pour ne pas donner au decodeur le niveau des patchs masques. Il n'y a pas de masque ici,
        donc mu porte sur toute la fenetre : c'est la meme quantite, estimee sur deux fois plus
        de points. sigma, lui, etait deja calcule sur la fenetre entiere au pretrain, et le
        plancher passe par _mask_stats : la convention est identique des deux cotes.
        """
        mu = x.mean(dim=2)                       # (B, F)
        sigma = x.std(dim=2, unbiased=False)     # (B, F)
        return self._mask_stats(mu, sigma)

    def forward(self, x):
        # x: (B, n_features, n_epochs) en entrée
        ## RevIN par fenetre, identique au pretrain : sans lui l'encodeur gele recoit un canal
        ## qui a garde l'offset et l'echelle de sa fenetre, alors qu'il n'a jamais vu que du
        ## centre reduit. inorm_mask a zero (RevIN inactif) laisse x inchange.
        mu, sigma = self._window_stats(x)
        x = (x - mu[..., None]) / sigma[..., None]
        out = self.encoder.forward(x) # (B, N + 1, embed_dim)
        z = torch.cat([out[:, 0], out[:, self.center_patch]], dim=-1) # (B, 2*embed_dim)
        z = self.dense2(self.activation(self.dense1(z))) # (B, n_outputs)
        return z
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self
    

def build_model(model_cfg, task_cfg, *, n_features, window_size, n_outputs=4):
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

    if model_cfg.name == "cnn_split": ## ne fait que du localizer
        if is_classifier:
            raise ValueError("cnn_split est un localizer : ses deux branches sont EW/NS, "
                             "ce qui n'a pas de sens pour la tâche de classification")
        return cnn_split(
            n_features,
            window_size,
            conv_layers=[tuple(layer) for layer in model_cfg.conv_layers],
            dense_layers=tuple(model_cfg.dense_layers),
            dropout=model_cfg.dropout,
            split_dense=model_cfg.split_dense,
            )
    
    if model_cfg.name == "small_vit":
        return small_vit(
            n_features,
            window_size,
            patch_size=model_cfg.patch_size,
            embed_dim=model_cfg.embed_dim,
            n_attn_heads=model_cfg.n_attn_heads,
            n_blocks=model_cfg.n_blocks,
            expansion_factor=model_cfg.expansion_factor,
            dropout_rate=model_cfg.dropout_rate
            ) 
    ## l'ancien nom reste accepte : des runs et des checkpoints le portent encore
    if model_cfg.name in ("maneuver_finetune", "maneuver_finetune_MAEv2") :
        return ManeuverFineTune(
            n_features=n_features,
            window_size=window_size,
            patch_size=model_cfg.patch_size,
            embed_dim=model_cfg.embed_dim,
            n_attn_heads=model_cfg.n_attn_heads,
            n_blocks=model_cfg.n_blocks,
            expansion_factor=model_cfg.expansion_factor,
            dropout_rate=model_cfg.dropout_rate,
            freeze_encoder=task_cfg.freeze_encoder,
            n_outputs=n_outputs
        )
    ### MODÈLES NON SUPERVISÉS : is_pretrain = True, on pretrain un backbone ViT sur données spacetrack avec masking inspiré de VideoMAE
    
    if model_cfg.name in ("miniMAE", "MAEv1", "MAEv2", "MAEv3"): 
        """
        Les modèles de pretraining non supervisés. On réutilisera les poids de l'encodeur
        """
        return TimeSeriesMAE(
            n_features,
            window_size,
            patch_size=model_cfg.patch_size,
            masking_ratio=model_cfg.masking_ratio,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_n_attn_heads=model_cfg.encoder_n_attn_heads,
            encoder_n_blocks=model_cfg.encoder_n_blocks,
            decoder_embed_dim= model_cfg.decoder_embed_dim,
            decoder_n_attn_heads= model_cfg.decoder_n_attn_heads,
            decoder_n_blocks=model_cfg.decoder_n_blocks,
            expansion_factor=model_cfg.expansion_factor,
            dropout_rate=model_cfg.dropout_rate
            )
    raise KeyError (f"modèle inconnu : {model_cfg.name}")

