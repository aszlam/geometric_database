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
       
        
    

