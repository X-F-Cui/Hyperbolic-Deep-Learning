## Standard libraries
import os
import numpy as np 
import random
import math
import json
from functools import partial

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import tensorflow as tf
print(tf.__version__)

# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    !pip install --quiet pytorch-lightning
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

#hyperbolic version linear layer
class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim]. If set to 0, then it is a normal lorentz linear layer.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 nonlin=None):
        super().__init__()
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)

        x = self.weight(self.dropout(x))
                      
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        if bias is not None:
            x = x + bias
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1.0) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features // self.in_features
        # with torch.no_grad():
        #     for idx in range(0, self.in_features, step):
        #         self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

class LorentzPositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 dropout=0.1):
        super(LorentzPositionwiseFeedForward, self).__init__()
        self.w_1 = LorentzLinear(d_model,
                                 d_ff,
                                 dropout=dropout)
        self.residual = LorentzLinear(d_ff, d_model, dropout=dropout, nonlin=nn.ReLU())

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        x_out = self.w_1(x)
        return self.residual(x_out, x)

    def update_dropout(self, dropout):
        pass

class LorentzMultiheadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = nn.Parameter(torch.tensor([math.sqrt(embed_dim)]))
        self.softmax = nn.Softmax(dim=-1)
        self.bias = nn.Parameter(torch.zeros(()))
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = LorentzLinear(input_dim, embed_dim)
        self.k_proj = LorentzLinear(input_dim, embed_dim)
        self.v_proj = LorentzLinear(input_dim, embed_dim)
        self.o_proj = LorentzLinear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight.weight)
        self.q_proj.weight.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight.weight)
        self.k_proj.weight.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight.weight)
        self.v_proj.weight.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight.weight)
        self.o_proj.weight.bias.data.fill_(0)
        
    def cinner(self, x: torch.Tensor, y: torch.Tensor):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
      d = u.size(dim) - 1
      uv = u * v
      if keepdim is False:
          return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
              dim, 1, d
          ).sum(dim=dim, keepdim=False)
      else:
          # return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
          #     dim=dim, keepdim=True
          # )
          return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
              dim=dim, keepdim=True
          )

    def mid_point(self, x, w=None):
        if w is not None:
            ave = w.matmul(x)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return ave / denom

    # Takes in x of shape (batch, seq, features)
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, input_dim = x.size()
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
              
        # Separate Q, K, V from linear output
        # qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        # qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        # q, k, v = qkv.chunk(3, dim=-1)
        
        # Determine value outputs
        attention = (2 + 2 * self.cinner(q, k)) / self.scale + self.bias
        attention = self.softmax(attention)

        values = self.mid_point(v, attention)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o

class LorentzLayerNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim-1)

    def forward(self, x):
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        x_narrow = self.norm(x_narrow)
        # compute time vector based on definition of hyperboloid model
        time = x_narrow.square().sum(2).add(1)
        time = time.sqrt().unsqueeze(-1)

        x = torch.cat([time, x_narrow], dim=-1)
        return x

class LorentzTransformerBlock(nn.Module):
    
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        self.self_attn = LorentzMultiheadAttention(input_dim, input_dim, num_heads)
        # feed-foward layer
        self.feed_forward = LorentzPositionwiseFeedForward(input_dim, dim_feedforward, dropout=dropout)
        self.residual = LorentzLinear(input_dim, input_dim, dropout=dropout, bias=False)
        # layer norm
        self.norm1 = LorentzLayerNorm(input_dim)
        self.norm2 = LorentzLayerNorm(input_dim)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        attn_out = self.residual(attn_out, x)
        attn_out = self.norm1(attn_out)
        # feed-forward part
        output = self.feed_forward(attn_out)  
        output = self.norm2(output)

        return output

class LorentzTransformer(nn.Module):
    
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([LorentzTransformerBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class LorentzPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.point = LorentzLinear(d_model, d_model)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        #x = x + self.pe[:, :x.size(1)]
        pe = self.pe[:x.size(0)]
        emb = self.point(x, pe)

        return emb

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
        
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class TransformerPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        # self.input_net = nn.Sequential(
        #     nn.Dropout(self.hparams.input_dropout),
        #     nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        # )
        self.input_net = LorentzLinear(self.hparams.input_dim, self.hparams.model_dim, dropout = 0.0)

        # Positional encoding for sequences
        self.positional_encoding = LorentzPositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = LorentzTransformer(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        # self.output_net = nn.Sequential(
        #     nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
        #     nn.LayerNorm(self.hparams.model_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.hparams.dropout),
        #     nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        # ) 
        self.output_net = nn.Sequential(LorentzLinear(self.hparams.model_dim, self.hparams.model_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hparams.model_dim, self.hparams.num_classes))

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer, 
                                             warmup=self.hparams.warmup, 
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError    

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

def processText(file):
    fileptr = open(file, "r")
    data = fileptr.read()
    # list of tokens
    tokens = data.split("\n")
    fileptr.close()
    
    num_words = len(set(tokens))
    tok_dict = {}
    word_ind = 0
    for word in tokens:
        if word in tok_dict:
            pass
        else:
            # plus one for later masked words
            vec = torch.zeros(num_words+1)
            vec[word_ind] = 1
            tok_dict[word] = vec
            word_ind += 1
    
    vec_list = [tok_dict[word] for word in tokens]
    # stack basis vector of each token together
    corpus = torch.stack(vec_list, 0)
            
    return corpus

class TextDataset(data.Dataset):

    def __init__(self, corpus, seq_len, size):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.size = size
        
        # list of sliced corpus matrix starting at a random row having seq_len rows
        self.data = [torch.narrow(self.corpus, 0, random.randint(0,len(self.corpus)-self.seq_len), self.seq_len) for _ in range(self.size)]
  
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        mask_ind = random.randint(0, self.seq_len-1)
        text_data = self.data[idx]
        
        # mask out the token at position mask_ind
        inp_data = text_data.clone()
        inp_data[mask_ind] = torch.zeros(text_data.shape[1])
        inp_data[mask_ind][-1] = 1
        
        return inp_data, text_data
    
class FillMaskPredictor(TransformerPredictor):
    
    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        # reverse one hot for labels
        labels = labels.transpose(1,2).argmax(dim=1)

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1,preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc
        
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

def train_mask(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "FillMaskTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir, 
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0, 
                         max_epochs=100,
                         gradient_clip_val=5)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "FillMaskTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = FillMaskPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = FillMaskPredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
        
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    
    model = model.to(device)
    return model, result

corpus = processText("HenryVIII_tok.txt")
dataset = partial(TextDataset, corpus, 20)
train_loader = data.DataLoader(dataset(5000), batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
val_loader   = data.DataLoader(dataset(2000), batch_size=32)
test_loader  = data.DataLoader(dataset(2000), batch_size=32)

mask_model, mask_result = train_mask(input_dim=train_loader.dataset.corpus.shape[1],
                                              model_dim=128,
                                              num_heads=1,
                                              num_classes=train_loader.dataset.corpus.shape[1],
                                              num_layers=1,
                                              dropout=0.0,
                                              lr=5e-4,
                                              warmup=50)

print(f"Val accuracy:  {(100.0 * mask_result['val_acc']):4.2f}%")
print(f"Test accuracy: {(100.0 * mask_result['test_acc']):4.2f}%")