from torch import nn

### on a en entrée un tenseur (B,F,W) = (batch_size, features, window_size).
## pour le moment on veut dim=2 en sortie : manoeuvre EW/NS ? 
class NaiveBaseLine(nn.Module):
    def __init__(self, n_features, window_size, n_outputs):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dim_embedding = n_features*window_size
        self.n_outputs= n_outputs
        self.layer1 = nn.Linear(self.dim_embedding, n_outputs) 

    def forward(self, x):
        batch_size, features, window_size = x.shape ## (B, F, W) 

        x = x.reshape(batch_size, self.dim_embedding) ## (B, F*W) = (B, dim_embedding)
        x = self.layer1(x) ## (B, 2)
        return x

## test
import torch 
def main():
    x = torch.randn(4,11,97) 
    testmodel = NaiveBaseLine(11, 97 , 2)
    y = testmodel(x)
    print(y.shape)

if __name__ == "__main__":
    main()




