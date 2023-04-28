import torch
import torch.nn as nn
import torch.nn.functional as F
import math


conv_hidden = 8
mlp_nfilters = 64
mlp_hidden = 256



# Generator model
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.num_channels = 3
        self.attn_hidden = 514
        self.mlp_nfilters = 32
        self.height_hidden = 8
        self.width_hidden = 12
        self.num_filters = 512
        self.s_dim = 384
        self.hidden_size = 512
        self.num_heads = 16
        
        
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
            nn.Conv2d(in_channels=256, out_channels=self.num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(True)
            # output is 512 x 8 x 12
        )
        
        x_coords = torch.linspace(-1, 1, self.width_hidden)
        y_coords = torch.linspace(-1, 1, self.height_hidden)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords)
        
        # Attention module
        self.image_query = nn.Linear(in_features=self.attn_hidden, out_features=self.s_dim)
        self.image_key = nn.Linear(in_features=self.attn_hidden, out_features=self.s_dim)
        self.image_value = nn.Linear(in_features=self.attn_hidden, out_features=self.s_dim)
        
        self.instruction_query = nn.Linear(in_features=self.s_dim, out_features=self.s_dim)
        self.instruction_key = nn.Linear(in_features=self.s_dim, out_features=self.s_dim)
        self.instruction_value = nn.Linear(in_features=self.s_dim, out_features=self.s_dim)
        
        self.linear_out = nn.Linear(self.s_dim+4, 32)
        
        
        # Image decoder
        self.decoder = nn.Sequential( # input is mlp_nfilters x 8 x 12
            nn.Conv2d(self.mlp_nfilters, 512, 3, 1, 1),
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
        phi_im_out = self.imencoder(x1)
        batch_size, num_channels, conv_h, conv_w = phi_im_out.size()
        n_pair = conv_h * conv_w
        x_grid = self.x_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        y_grid = self.y_grid.reshape(1, 1, conv_h, conv_w).repeat(batch_size, 1, 1, 1)
        coord_tensor = torch.cat((x_grid, y_grid), dim=1)
        if torch.cuda.is_available():
            coord_tensor = coord_tensor.cuda()
        phi_im = torch.cat([phi_im_out, coord_tensor], dim=1)
        phi_im = phi_im.view(batch_size, n_pair, num_channels+2)

        # Sentence embedding
        phi_s = sen_embed.view(-1, sen_embed.shape[-1])
        
        # Attention module
        im_query = self.image_query(phi_im)
        im_key = self.image_key(phi_im)
        im_value = self.image_value(phi_im)
        
        sen_query = self.instruction_query(phi_s)
        sen_key = self.instruction_key(phi_s)
        sen_value = self.instruction_value(phi_s)
        # Reshape tensors
        seq_len = im_query.size(1)
        im_query = im_query.view(batch_size, self.s_dim, -1)
        im_key = im_key.view(batch_size, self.s_dim, -1)
        im_value = im_value.view(batch_size, self.s_dim, -1)
        seq_len = sen_query.size(1)
        sen_query = sen_query.view(batch_size, seq_len, -1)
        sen_key = sen_key.view(batch_size, seq_len, -1)
        sen_value = sen_value.view(batch_size, seq_len, -1)
        # Compute scores
        im_scores = torch.matmul(im_query, im_key.transpose(1,2))
        sen_scores = torch.matmul(sen_query, sen_key.transpose(1,2))
        # normalize to get weights
        im_weights = im_scores / ((self.s_dim // self.num_heads)**0.5)
        sen_weights = sen_scores / ((self.s_dim // self.num_heads)**0.5)
        # Compute output
        im_out = torch.matmul(im_weights, im_value)
        sen_out = torch.matmul(sen_weights, sen_value)
        # Conncatenate tensors
        phi = torch.cat([im_out, sen_out], dim=2)
        phi = phi.view(batch_size, -1, 8, 12)
        # Transform multi-head attention
        phi = self.linear_out(phi.transpose(1,3)).view(batch_size, self.mlp_nfilters, conv_h, conv_w)
        
        # Decoder
        x = self.decoder(phi)
        
        # reshape phi_im tensor and phi_s tensor
        phi_im = phi_im_out.view(batch_size, self.num_filters, conv_h, conv_w)
        phi_s = phi_s.unsqueeze(dim=2)
        phi_s = phi_s.unsqueeze(dim=2)
        
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
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is (ndf*8) x 8 x 8
        
        self.phiencoder = nn.Sequential( # input is (mlp_nfilters) x 8 x 8
            nn.Conv2d(self.mlp_nfilters, self.mlp_nfilters*2, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.mlp_nfilters*2, self.mlp_nfilters*4, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.mlp_nfilters*4, self.mlp_nfilters*8, 3, 1, 1),
            nn.BatchNorm2d(self.mlp_nfilters*8),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is (mlp_nfilters*8) x 8 x 8
        
        self.phiimencoder = nn.Sequential( # input is (ngf*8) x 8 x 8
            nn.Conv2d(self.ngf*8, self.ndf*8, 3, 1, 1), # (ngf*8) x 8 x 8
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is # (ndf*8) x 8 x 8

        self.phisencoder = nn.Sequential( # input is (2*lstm_hidden) x 1 x 1
            nn.ConvTranspose2d(2*self.lstm_hidden, self.ndf*4, 4, 1, 1, bias=False), # (ndf*8) x 4 x 4
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.ndf*4, self.ndf*2, 4, 1, 1, bias=False),  # (ndf*8) x 8 x 8
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.ndf*2, self.ndf*8, 4, (2,4), 0, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ) # output is # (ndf*8) x 8 x 8

        self.classifier = nn.Sequential( #input is (n_filt*8) x 8 x 8
            nn.Conv2d(self.n_filt*8, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, 1, 4, (1,3), 0, bias=False),
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