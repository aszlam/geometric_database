import math
import torch
import torch.nn as nn

#"what you see" field vs "what is here" field


class ResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bnorms = nn.ModuleList([nn.BatchNorm1d(out_d),
                                         nn.BatchNorm1d(out_d)])
        self.linears = nn.ModuleList([nn.Linear(in_d, out_d),
                                      nn.Linear(out_d, out_d)])
        if in_d != out_d:
            self.proj = nn.Linear(in_d, out_d)
            
        self.relu = nn.ReLU()
    
    def forward(self, x):        
        z = self.linears[0](x)
        if self.use_batchnorm:
            z = self.bnorms[0](z)
        z = self.relu(z)
        z = self.linears[1](z)
        if self.use_batchnorm:
            z = self.bnorms[1](z)
        z = self.relu(z)
        if getattr(self, "proj", None):
            u = self.proj(x)
        else:
            u = x
        return z + u


class PosToFeature(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        d = opts["embedding_dim"]
        if opts.get("fourier_embedding_dim") and opts["fourier_embedding_dim"] > 0:
            ff = FourierFeatures(opts)
            proj = nn.Linear(opts["fourier_embedding_dim"]*2, d)
            self.embedding = nn.Sequential(ff, proj)
        else:
            self.embedding = nn.Linear(2, d)
        self.relu = nn.ReLU()
        layers = [ResidualBlock(d, d, use_batchnorm=opts.get("use_batchnorm", False))
                  for i in range(opts["num_layers"])]
        self.layers = nn.ModuleList(layers)

    def forward(self, pos):
        z = self.relu(self.embedding(pos))
        for l in self.layers:
            z = l(z)
        return z


class PoseToScalar(nn.Module):
    def __init__(self, opts):
        super().__init__()
        d = opts["embedding_dim"]
        if opts.get("fourier_embedding_dim") and opts["fourier_embedding_dim"] > 0:
            ff = FourierFeatures(opts)
            proj = nn.Linear(opts["fourier_embedding_dim"]*2, d)
            self.embedding = nn.Sequential(ff, proj)
        else:
            self.embedding = nn.Linear(2, d)
        self.combiner = nn.Linear(2*d, d)
        layers = [ResidualBlock(d, d, use_batchnorm=opts.get("use_batchnorm", False))
                  for i in range(opts["num_layers"])]
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(d, 1)
        
        
    def forward(self, pose, scene_pos_embedding):
        """
        pose is batchsize x 2
        feature dim is a, b
        scene_pos_embedding is batchsize by d output of a PosToFeature map
        """
        z = torch.cat([self.embedding(pose), scene_pos_embedding], 1)
        z = self.combiner(z)
        for l in self.layers:
            z = l(z)
            
        return self.out(z)
       
        
class FourierFeatures(nn.Module):
    def __init__(self, opts):
        super().__init__()
        d = opts["fourier_embedding_dim"]
        alpha = opts["fourier_embedding_scale"]
        self.B = 2*math.pi*alpha * torch.randn(2, d)
        self.B.requires_grad = False

    def forward(self, x):
        """
        assumes B x 2 input
        """
        return torch.cat([torch.cos(x@self.B), torch.sin(x@self.B)], 1) 

    def cuda(self, device=None):
        self.B = self.B.cuda(device=device)
        self.B.requires_grad = False
        return self
        
    def cpu(self):
        self.B = self.B.cpu()
        self.B.requires_grad = False
        return self
