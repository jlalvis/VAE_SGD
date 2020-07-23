#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:39:49 2019

@author: jorge
"""
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 256, 6, 2)

class VAE(nn.Module):
    def __init__(self, gpath, image_channels=1, h_dim=12*256, z_dim=20, cuda=True):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(32), #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(64), #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128), #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(256), #nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten(),
            nn.Linear(12*256,512),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(512, z_dim)
        self.fc2 = nn.Linear(512, z_dim)
        self.fc3 = nn.Linear(z_dim, 512)
        
        self.decoder = nn.Sequential(
            nn.Linear(512,12*256),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(128), #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(64), #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(32), #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2, bias=False),
            nn.Sigmoid(),
        )
        if cuda:
            self.load_state_dict(torch.load(gpath))
        else:
            self.load_state_dict(torch.load(gpath, map_location=lambda storage, loc: storage))
        
    def reparameterize(self, mu, logvar):
        # for generation no noise is added (mean autoencoder is used)
        #std = logvar.mul(0.5).exp_()
        #esp = torch.randn(mu.size())
        z = mu #+ std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar
