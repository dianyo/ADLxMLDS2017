
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, n_filter=256):
        super(Generator, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(23, 100),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 8 *8),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 512, 5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 3, 1),
            nn.Tanh(),
        )
    def forward(self, input, c1, c2):
        c = torch.cat((c1.view(-1, 12),c2.view(-1, 11)), 1)
        c = self.embed(c)
        x = input * c
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.dc = nn.Sequential(
            nn.Linear(512, 1),
        )
        self.c1 = nn.Sequential(
            nn.Linear(512, 12),
        )
        
        self.c2 = nn.Sequential(
            nn.Linear(512, 11),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*4*4)
        x = self.fc1(x)
        d = self.dc(x)
        c_1 = self.c1(x)
        c_2 = self.c2(x)

        return d, c_1, c_2

