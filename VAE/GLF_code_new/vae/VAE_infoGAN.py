import torch.nn as nn
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
class ConvVAE(nn.Module):
    def __init__(self, args):
        super(ConvVAE, self).__init__()
        if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
            self.nc = 1
        else:
            self.nc = 3
        self.args = args
        self.have_cuda = True
        self.nz = args.num_latent
        self.input_size = args.image_size
        self.conv = nn.Sequential(
            nn.Conv2d(self.nc, 64, 4, 2, 1),            
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.encode_fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.nz),
        )
        
        self.decode_fc = nn.Sequential(
            nn.Linear(self.nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.nc, 4, 2, 1),
            nn.Tanh(),
        )
        
        initialize_weights(self)

    def encode(self, x):
        conv = self.conv(x)
        distributions = self.encode_fc(conv.view(-1, 128*(self.input_size//4)*(self.input_size//4)))
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        latent = reparametrize(mu, logvar)
        return latent, mu, logvar
    
    def decode(self, z):
        deconv_input = self.decode_fc(z)
        deconv_input = deconv_input.view(-1,128, self.input_size//4, self.input_size//4)
        output = self.deconv(deconv_input)
        if not self.args.loss_type == 'Perceptual':
            output = output*0.5 + 0.5
        return output
            
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        decoded = self.decode(z)
        return decoded, z, mu, logvar