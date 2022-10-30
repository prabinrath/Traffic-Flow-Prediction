class FusionWeight(nn.Module):
    def __init__(self, map_dim):
        super().__init__()
        # 1 dimension for batch processing, this class cannot process unbatched data
        self.w = nn.Parameter(torch.randn(1, *map_dim))
        
    def forward(self, y):
        y *= self.w # Eliment wise product (Hadamard Product)
        return y
    
class STResnet(nn.Module):
    def __init__(self, c_channel, p_channel, t_channel, n_residual_units, map_dim, use_bn = False):
        super().__init__()
        self.c_pipe = ResidualPipeline(c_channel, n_residual_units, use_bn)
        self.p_pipe = ResidualPipeline(p_channel, n_residual_units, use_bn)
        self.t_pipe = ResidualPipeline(t_channel, n_residual_units, use_bn)        
        self.fuse_c = FusionWeight(map_dim)
        self.fuse_p = FusionWeight(map_dim)
        self.fuse_t = FusionWeight(map_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, x_c, x_p, x_t):
        y_c = self.c_pipe(x_c)
        y_p = self.p_pipe(x_p)
        y_t = self.t_pipe(x_t)        
        y = self.fuse_c(y_c) + self.fuse_p(y_p) + self.fuse_t(y_t) # Fusion
        y = self.tanh(y)
        return y

##########################################################################################################################

class STResnet(nn.Module):
    def __init__(self, c_channel, p_channel, t_channel, n_residual_units, map_dim, use_bn = False):
        super().__init__()
        self.c_pipe = ResidualPipeline(c_channel, n_residual_units, use_bn)
        self.p_pipe = ResidualPipeline(p_channel, n_residual_units, use_bn)
        self.t_pipe = ResidualPipeline(t_channel, n_residual_units, use_bn)
        # 1 dimension for batch processing, this class cannot process unbatched data
        self.w_c = nn.Parameter(torch.randn(1, *map_dim))
        self.w_p = nn.Parameter(torch.randn(1, *map_dim))
        self.w_t = nn.Parameter(torch.randn(1, *map_dim))
        self.tanh = nn.Tanh()
        
    def forward(self, x_c, x_p, x_t):
        y_c = self.c_pipe(x_c)
        y_p = self.p_pipe(x_p)
        y_t = self.t_pipe(x_t)
        # Fusion
        y = self.w_c*y_c + self.w_p*y_p + self.w_t*y_t # Eliment wise product (Hadamard Product)
        y = self.tanh(y)
        return y
		
##########################################################################################################################

class FusionWeight(nn.Module):
    def __init__(self, map_dim):
        super().__init__()
        # 1 dimension for batch processing, this class cannot process unbatched data
        self.w = nn.Parameter(torch.randn(1, *map_dim))
        
    def forward(self, y):
        y *= self.w # Eliment wise product (Hadamard Product)
        return y
    
class STResnet(nn.Module):
    def __init__(self, c_channel, p_channel, t_channel, n_residual_units, map_dim, use_bn = False):
        super().__init__()
        self.n_residual_units = n_residual_units
        self.map_dim = map_dim
        self.use_bn = use_bn        
        self.fuse_c = self.build_pipeline(c_channel)
        self.fuse_p = self.build_pipeline(p_channel)
        self.fuse_t = self.build_pipeline(t_channel)
        self.tanh = nn.Tanh()
    
    def build_pipeline(self, in_channels):
        return nn.Sequential(OrderedDict([
            ('res_pipe', ResidualPipeline(in_channels, self.n_residual_units, self.use_bn)),
            ('fuse', FusionWeight(self.map_dim)),
        ]))
        
    def forward(self, x_c, x_p, x_t):      
        y = self.fuse_c(x_c) + self.fuse_p(x_p) + self.fuse_t(x_t) # Fusion
        y = self.tanh(y)
        return y