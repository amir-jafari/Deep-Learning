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
    This is the documentation for the PatcherUnfold class.

    Class: PatcherUnfold(nn.Module)

    Description:
        The PatcherUnfold class is a module for unfolding an input tensor into patches, performing linear projection on the patches, and returning the transformed patches.

    Attributes:
        - unfold: nn.Unfold
            Object that unfolds the input tensor into patches based on kernel_size and stride.
        - linear_projection: nn.Linear
            Object that performs linear projection on the unfolded patches.

    Methods:
        - __init__()
            Constructor method that initializes the PatcherUnfold class.

        - forward(x)
            Method that performs the forward pass of the PatcherUnfold class.

    Note:
        - This class inherits from the nn.Module class.

    Method Details:
        1. __init__()

            Description:
                Initializes the PatcherUnfold class by defining the unfold and linear_projection attributes.

            Parameters:
                None

            Returns:
                None

        2. forward(x)

            Description:
                Performs the forward pass of the PatcherUnfold class by applying the unfolding, permuting, and linear projection operations.

            Parameters:
                - x: torch.Tensor
                    Input tensor to be transformed.

            Returns:
                torch.Tensor
                    Transformed tensor after unfolding and linear projection operations.
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
    This module provides the implementation of the PatcherConv class, which is a PyTorch module for patch-based convolution operations.

    PatcherConv Class:
        This class extends the nn.Module class from the PyTorch library. It performs patch-based convolution operations on the input tensor.

    Constructor:
        def __init__(self)
            Initializes an instance of the PatcherConv class.

            Parameters:
                None

            Returns:
                None

    Attributes:
        conv: nn.Conv2d
            A convolutional layer that applies patch-based convolution to the input tensor.

        flatten: nn.Flatten
            A flatten layer that flattens the output tensor after the convolution operation.

    Methods:
        forward(self, x)
            Performs the forward pass of the PatcherConv module.

            Parameters:
                x: torch.Tensor
                    The input tensor of shape [batch_size, channels, height, width].

            Returns:
                torch.Tensor
                    The output tensor of shape [batch_size, num_patches, embedding_dim].
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
    This module implements the PosEmbedding class for positional embedding.

    Attributes:
        - class_token (torch.Tensor): A learnable parameter representing the class token with shape (BATCH_SIZE, 1, EMBED_DIM).
        - pos_embedding (torch.Tensor): A learnable parameter representing the positional embedding with shape (BATCH_SIZE, NUM_PATCHES+1, EMBED_DIM).
        - dropout (torch.nn.Dropout): Dropout layer for regularization.

    Methods:
        - forward(x): Performs the forward pass of the PosEmbedding module.

        Example usage:
            >>> pos_embedding = PosEmbedding()
            >>> output = pos_embedding.forward(x)
            >>> print(output.size())
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
    This module defines the MLPHead class, which is a subclass of nn.Module.

    class MLPHead(nn.Module):
        def __init__(self):
            super().__init__()

            self.norm = nn.LayerNorm(EMBED_DIM)
            self.linear_0 = nn.Linear(in_features=EMBED_DIM, out_features=EMBED_DIM)
            self.gelu = nn.GELU()
            self.linear_1 = nn.Linear(in_features=EMBED_DIM, out_features=NUM_CLASSES)


        def forward(self, x):
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
    This code represents a class called "ViT" which is a custom implementation of the Vision Transformer model.

    The class inherits from the nn.Module class from the PyTorch framework, which allows it to be used as a module in a larger neural network architecture.

    The "__init__" method initializes the class and defines its components.

    - The "patcher" attribute is an instance of the "PatcherConv" class, which is responsible for patching the input image.
    - The "pos_embedding" attribute is an instance of the "PosEmbedding" class, which adds positional embeddings to the patches.
    - The "transformer_encoders" attribute is an nn.Sequential container that holds a series of "TransformerEncoder" instances. The number of encoder instances is determined by the "NUM_ENCODERS" constant.
    - The "mlp_head" attribute is an instance of the "MLPHead" class, which represents the MLP-based classification head of the model.

    The "forward" method performs the forward pass of the model.

    - The input tensor "x" is patched using the "patcher" attribute.
    - The positional embeddings are added to the patches using the "pos_embedding" attribute.
    - The patches are then passed through the transformer encoders using the "transformer_encoders" attribute.
    - Only the first element of the resulting tensor along the dimension 0 is selected using indexing.
    - Finally, the selected tensor is passed through the MLP head using the "mlp_head" attribute.

    The output of the "forward" method is the final output of the model.

    Note: This documentation assumes that the imported modules and constants are properly defined.
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