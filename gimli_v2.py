import os
import torch
import torch.nn as nn


conv_hidden = 8
mlp_nfilters = 64
mlp_hidden = 256



# Generator model
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.num_channels = 3
        self.mlp_hidden = 256
        self.mlp_nfilters = 32
        self.lstm_bidirectional = True
        self.height_hidden = 4
        self.width_hidden = 6
        
        
        # Image encoder
        self.imencoder = nn.Sequential( # input is (nc) x 120 x 180
            nn.Conv2d(in_channels=self.num_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=3, bias=False), # changed to p=3 
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
            # output is 1024 x 4 x 6
        )
        
        x_coords = torch.linspace(-1, 1, self.width_hidden)
        y_coords = torch.linspace(-1, 1, self.height_hidden)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords)
        
                
        # Linear transform (y = xA^T + b)
        self.g = nn.Sequential(
            nn.Linear(2436, self.mlp_hidden),
            nn.ReLU(True),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(True),
            nn.Linear(self.mlp_hidden, self.height_hidden * self.width_hidden * self.mlp_nfilters),
            nn.ReLU(True)
        )
        
        # Image decoder
        self.decoder = nn.Sequential( # input is mlp_nfilters x 4 x 6
            nn.Conv2d(self.mlp_nfilters, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.num_channels, 4, 2, (1, 3), bias=False),
            nn.Tanh()
        )
        
    def forward(self, x1, sen_embed):
        
        # Image encoder
        phi_im = self.imencoder(x1)
        batch_size, n_channel, conv_h, conv_w = phi_im.size()
        n_pair = conv_h * conv_w
        # Cast all pairs against each other
        x_grid = self.x_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        y_grid = self.y_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        coord_tensor = torch.cat((x_grid, y_grid), dim=1)
        if torch.cuda.is_available():
            coord_tensor = coord_tensor.cuda()
        x1 = torch.cat([phi_im, coord_tensor], 1) # (B x 512+2 x 8 x 8)
        x1 = x1.permute(0, 2, 3, 1).view(batch_size, conv_h * conv_w, n_channel+2)
        x_i = torch.unsqueeze(x1, 1) # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, n_pair, 1, 1) # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x1, 2) # (B x 64 x 1 x 26)
        x_j = x_j.repeat(1, 1, n_pair, 1) # (B x 64 x 64 x 2*26)
        x1 = torch.cat([x_i, x_j], 3)

        x2 = sen_embed
        phi_s = torch.unsqueeze(x2, 2)
        phi_s = torch.unsqueeze(phi_s, 3)
        x2 = torch.unsqueeze(x2, 1)
        x2 = x2.repeat(1, n_pair, 1)
        x2 = torch.unsqueeze(x2, 2)
        x2 = x2.repeat(1, 1, n_pair, 1)

        # Relational module
        phi = torch.cat([x1, x2], 3)
        
        # Changed n_concat from 1412 to 2436
        phi = phi.view(batch_size * (n_pair**2), 2436)
        phi = self.g(phi)
        phi = phi.view(batch_size, n_pair**2, n_pair*self.mlp_nfilters).sum(1)
        phi = phi.view(batch_size, self.mlp_nfilters, conv_h, conv_w)
        
        # Decoder
        x = self.decoder(phi)
        
        return x, phi, phi_im, phi_s
    
    
# Discriminator model
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.num_channels = 3
        self.ndf = 64
        self.ngf = 64
        self.mlp_nfilters = 32
        self.lstm_hidden = 192
        self.n_filt = self.ndf + self.ndf + self.mlp_nfilters + self.ndf
        
        self.imencoder = nn.Sequential( # input is (num_channels) x 128 x 128
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 3, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        ) # output is (ndf*8) x 4 x 6
        
        self.phiencoder = nn.Sequential( # input is (mlp_nfilters) x 8 x 8
            nn.Conv2d(self.mlp_nfilters, self.mlp_nfilters*2, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.mlp_nfilters*2, self.mlp_nfilters*4, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.mlp_nfilters*4, self.mlp_nfilters*8, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.mlp_nfilters*8, self.mlp_nfilters*16, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*16),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is (mlp_nfilters*8) x 4 x 6
        
        self.phiimencoder = nn.Sequential( # input is (ngf*8) x 8 x 8
            nn.Conv2d(self.ngf*16, self.ngf*8, 3, 1, 1),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ngf*8, self.ndf*8, 3, 1, 1), # (ngf*8) x 8 x 8
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is # (ndf*8) x 4 x 6

        self.phisencoder = nn.Sequential( # input is (2*lstm_hidden) x 1 x 1
            nn.ConvTranspose2d(2*self.lstm_hidden, self.ndf*4, 4, 1, 1, bias=False), # (ndf*8) x 4 x 4
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.ndf*4, self.ndf*2, 4, 1, 1, bias=False),  # (ndf*8) x 8 x 8
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.ndf*2, self.ndf*8, 4, (1,2), 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is # (ndf*8) x 4 x 6

        self.classifier = nn.Sequential( #input is (n_filt*8) x 4 x 6
            nn.Conv2d(2560, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, 1, 4, (1,3), 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x1, phi, phi_im, phi_s):
        
        x1_embed = self.imencoder(x1)
        phi_embed = self.phiencoder(phi)
        phi_im_embed = self.phiimencoder(phi_im)
        phi_s_embed = self.phisencoder(phi_s)
        x = torch.cat([x1_embed, phi_embed, phi_im_embed, phi_s_embed], 1)
        x = self.classifier(x)
        
        return x, x1_embed
    
# Randomly initialize weights to mean = 0, std = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )
    
# MLP for g in RN for generator and discriminator
def g(n_concat):
    return nn.Sequential(
        nn.Linear(n_concat, mlp_hidden),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(mlp_hidden, mlp_hidden),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(mlp_hidden, conv_hidden * conv_hidden * mlp_nfilters),
        nn.LeakyReLU(0.2, inplace=True)
    )
    
    