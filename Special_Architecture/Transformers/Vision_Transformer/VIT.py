#%%
# Codeblock 1
import torch
import torch.nn as nn
from torchinfo import summary
#%%
# Codeblock 2
#(1)
BATCH_SIZE   = 1
IMAGE_SIZE   = 224
IN_CHANNELS  = 3

#(2)
PATCH_SIZE   = 16
NUM_HEADS    = 12
NUM_ENCODERS = 12
EMBED_DIM    = 768
MLP_SIZE     = EMBED_DIM * 4    # 768*4 = 3072

#(3)
NUM_PATCHES  = (IMAGE_SIZE//PATCH_SIZE) ** 2    # (224//16)**2 = 196

#(4)
DROPOUT_RATE = 0.1
NUM_CLASSES  = 10
#%%
# Codeblock 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#%%
# Codeblock 4
class PatcherUnfold(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)    #(1)
        self.linear_projection = nn.Linear(in_features=IN_CHANNELS*PATCH_SIZE*PATCH_SIZE, 
                                           out_features=EMBED_DIM)    #(2)
# Codeblock 5
    def forward(self, x):
        print(f'original\t: {x.size()}')
        
        x = self.unfold(x)
        print(f'after unfold\t: {x.size()}')
        
        x = x.permute(0, 2, 1)    #(1)
        print(f'after permute\t: {x.size()}')
        
        x = self.linear_projection(x)
        print(f'after lin proj\t: {x.size()}')
        
        return x
#%%
# Codeblock 6
patcher_unfold = PatcherUnfold()
x = torch.randn(1, 3, 224, 224)
x = patcher_unfold(x)
#%%
# Codeblock 7
class PatcherConv(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=IN_CHANNELS, 
                              out_channels=EMBED_DIM, 
                              kernel_size=PATCH_SIZE, 
                              stride=PATCH_SIZE)
        
        self.flatten = nn.Flatten(start_dim=2)
    
    def forward(self, x):
        print(f'original\t\t: {x.size()}')
        
        x = self.conv(x)    #(1)
        print(f'after conv\t\t: {x.size()}')
        
        x = self.flatten(x)    #(2)
        print(f'after flatten\t\t: {x.size()}')
        
        x = x.permute(0, 2, 1)    #(3)
        print(f'after permute\t\t: {x.size()}')
        
        return x
#%%
# Codeblock 8
patcher_conv = PatcherConv()
x = torch.randn(1, 3, 224, 224)
x = patcher_conv(x)
#%%
# Codeblock 9
class PosEmbedding(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(size=(BATCH_SIZE, 1, EMBED_DIM)), 
                                        requires_grad=True)    #(1)
        self.pos_embedding = nn.Parameter(torch.randn(size=(BATCH_SIZE, NUM_PATCHES+1, EMBED_DIM)), 
                                          requires_grad=True)    #(2)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)  #(3)

# Codeblock 10
    def forward(self, x):
        
        class_token = self.class_token
        print(f'class_token dim\t\t: {class_token.size()}')
        
        print(f'before concat\t\t: {x.size()}')
        x = torch.cat([class_token, x], dim=1)    #(1)
        print(f'after concat\t\t: {x.size()}')
        
        x = self.pos_embedding + x    #(2)
        print(f'after pos_embedding\t: {x.size()}')
        
        x = self.dropout(x)    #(3)
        print(f'after dropout\t\t: {x.size()}')
        
        return x
#%%
# Codeblock 11
pos_embedding = PosEmbedding()
x = pos_embedding(x)
#%%
# Codeblock 12
class TransformerEncoder(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        
        self.norm_0 = nn.LayerNorm(EMBED_DIM)    #(1)
        
        self.multihead_attention = nn.MultiheadAttention(EMBED_DIM,    #(2) 
                                                         num_heads=NUM_HEADS, 
                                                         batch_first=True, 
                                                         dropout=DROPOUT_RATE)
        
        self.norm_1 = nn.LayerNorm(EMBED_DIM)    #(3)
        
        self.mlp = nn.Sequential(    #(4)
            nn.Linear(in_features=EMBED_DIM, out_features=MLP_SIZE),    #(5)
            nn.GELU(), 
            nn.Dropout(p=DROPOUT_RATE), 
            nn.Linear(in_features=MLP_SIZE, out_features=EMBED_DIM),    #(6) 
            nn.Dropout(p=DROPOUT_RATE)
        )
        
# Codeblock 13
    def forward(self, x):
        
        residual = x    #(1)
        print(f'residual dim\t\t: {residual.size()}')
        
        x = self.norm_0(x)    #(2)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.multihead_attention(x, x, x)[0]    #(3)
        print(f'after attention\t\t: {x.size()}')
        
        x = x + residual    #(4)
        print(f'after addition\t\t: {x.size()}')
        
        residual = x    #(5)
        print(f'residual dim\t\t: {residual.size()}')
        
        x = self.norm_1(x)    #(6)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.mlp(x)    #(7)
        print(f'after mlp\t\t: {x.size()}')
        
        x = x + residual    #(8)
        print(f'after addition\t\t: {x.size()}')
        
        return x
#%%
# Codeblock 14
transformer_encoder = TransformerEncoder()
x = transformer_encoder(x)
#%%
# Codeblock 15
class MLPHead(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.linear_0 = nn.Linear(in_features=EMBED_DIM, 
                                  out_features=EMBED_DIM)
        self.gelu = nn.GELU()
        self.linear_1 = nn.Linear(in_features=EMBED_DIM, 
                                  out_features=NUM_CLASSES)    #(1)
        
    def forward(self, x):
        print(f'original\t\t: {x.size()}')
        
        x = self.norm(x)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.linear_0(x)
        print(f'after layer_0 mlp\t: {x.size()}')
        
        x = self.gelu(x)
        print(f'after gelu\t\t: {x.size()}')
        
        x = self.linear_1(x)
        print(f'after layer_1 mlp\t: {x.size()}')
        
        return x
#%%
# Codeblock 16
x = x[:, 0]    #(1)
mlp_head = MLPHead()
x = mlp_head(x)
#%%
# Codeblock 17
class ViT(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
    
        #self.patcher = PatcherUnfold()
        self.patcher = PatcherConv()    #(1) 
        self.pos_embedding = PosEmbedding()
        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder() for _ in range(NUM_ENCODERS)]    #(2)
            )
        self.mlp_head = MLPHead()
    
    def forward(self, x):
        
        x = self.patcher(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoders(x)
        x = x[:, 0]    #(3)
        x = self.mlp_head(x)
        
        return x
#%%
# Codeblock 18
vit = ViT().to(device)
x = torch.randn(1, 3, 224, 224).to(device)
print(vit(x).size())
#%%
# Codeblock 19
summary(vit, input_size=(1,3,224,224))