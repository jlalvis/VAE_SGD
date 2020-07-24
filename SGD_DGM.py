#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:30:37 2020

@author: jorge
"""
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from dataclasses import dataclass
from scipy.stats import chi
import pygimli as pg

@dataclass
class SGDsetup:
    epochs: int = 1
    batch_size: int = 1
    lr: float = 1e-4
    clr: float = 1.0
    ninits: int = 1
    zinit: np.ndarray = np.zeros(1)
    gen: nn.Module = None
    fwd: str = ''
    A: np.ndarray = np.zeros(1)
    tt: str = None
    d: np.ndarray = np.zeros(1)
    #clip: str = ''
    reg: str = ''
    #alpha: float = 0.0
    lam: float = 0.0
    clam: float = 1.0
    device: str = ''
    step_eps: int = 1
    DGM: str = ''
    optimizer: str = 'SGD'
    truemodel: np.ndarray = np.zeros(1)

def SGD_ninits(setup):
    """Perform stochastic gradiend descent for a number of initial models.
    Arguments:
    setup -- SGDsetup object which contains SGD parameters.
    """
    print_SGDsetup(setup)
    dRMSEs = []
    models_costmin = []
    zs_hist = []
    mRMSEs = []
    for j in range(setup.ninits):
        # call SGD_DGM_linear with one initial model:
        zi = setup.zinit[j]
        dloss, mloss, modelmin, zs = SGD_DGM(setup,zi)
        dRMSEs.append(np.sqrt(np.array(dloss)/len(setup.d)))
        models_costmin.append(modelmin)
        zs_hist.append(zs)
        mRMSEs.append(mloss)
        minit = np.argmin(dRMSEs[-1])
        print('init: {0}, dRMSE: {1:.4f}, mRMSE: {2:.4f}, ||z||: {3:.4f}'.format(j,
              dRMSEs[-1][minit], mRMSEs[-1][minit], np.linalg.norm(zs_hist[-1][minit])))

    return dRMSEs,mRMSEs,models_costmin,zs_hist

def SGD_DGM(setup,zi):
    """Perform stochastic gradiend descent for one model and a linear forward operator.
    Arguments:
    setup -- SGDsetup object which contains SGD parameters.
    zi -- initial model as np.ndarray.
    """
    #optimizer = optim.Adam([z], lr=lr)
    nz = setup.zinit.shape[1]
    if setup.DGM == 'VAE':
        z = 1.0*torch.zeros([1, nz]).to(setup.device)
        z.data = torch.Tensor(zi.reshape(1,-1)).to(setup.device)

    if setup.DGM == 'SGAN':
        nzx = 5
        nzy = 3
        z = 1.0*torch.zeros([1,1,nzx,nzy]).to(setup.device)
        z.data = torch.Tensor(zi.reshape(1,1,nzx,nzy)).to(setup.device)

    z.requires_grad = True
    ndata = len(setup.d)
    nz = z.numel()
    lam = setup.lam
    sg_its = ndata//setup.batch_size
    truemodel = setup.truemodel
    dloss = []
    zs = []
    mloss = []
    # set torch optimizer:
    if setup.optimizer == "SGD":
        optimizer = optim.SGD([z], lr=setup.lr)#, momentum=0.5)
    elif setup.optimizer == 'Adam':
        optimizer = optim.Adam([z], lr=setup.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=setup.step_eps,gamma=setup.clr)
    # send computation vectors to device:
    d = torch.Tensor(setup.d).to(setup.device)
    A = torch.Tensor(setup.A).to(setup.device)
    tt = setup.tt
    chidist = chi(nz)
    mchi = chidist.mean()
    for it in range(setup.epochs):
        mask = np.ones(ndata, dtype=bool)
        inds_init = np.arange(0,ndata)
        inds_left = np.copy(inds_init)
        for sg in range(sg_its):
            # get data batch indices:
            rand_outs = np.random.choice(len(inds_left),setup.batch_size,replace=False)
            inds_batch = inds_left[rand_outs]
            mask[inds_batch] = False
            inds_left = inds_init[mask]

            # stochastic clipping for SGAN if z is outside range "box"
            if setup.DGM=='SGAN':
                maskz = torch.zeros(z.shape,dtype=bool)
                maskz[(z.data > 1) | (z.data < -1)] = 1
                z.data[maskz] = torch.randn(z.shape)[maskz].to(setup.device)

            # Produce model from z
            x0 = setup.gen(z)
            if setup.DGM=='SGAN':
                x0 = (x0 + 1) * 0.5

            x=np.copy(x0.data.cpu().numpy()) # copy x into numpy array.

            # Compute cost and gradient on model
            #s_model = x0
            s_model = 0.06 + 0.02*(1-x0)
            s_model=1/s_model

            if setup.fwd == 'nonlinear':
                s_m = s_model.detach().cpu().numpy()
                tt.velocity = pg.Vector(np.float64(1./s_m.flatten()))
                tt.fop.createJacobian(1./tt.velocity)
                J = np.zeros((ndata,tt.inv.parameterCount))
                for i in range(ndata):
                    J[i,:] = np.array(tt.fop.jacobian().row(i))
                J = torch.Tensor(J).to(setup.device)
                Jsgd = J[inds_batch]
                A = J.to(setup.device)

            sim=torch.mv(A,s_model.flatten())
            e=d[inds_batch]-sim[inds_batch]
            databatch_loss = torch.sum(e**2)
            optimizer.zero_grad()

            if setup.fwd == 'linear':
                databatch_loss.backward()
            elif setup.fwd == 'nonlinear':
                grad = torch.mv(-Jsgd.T,e.T)
                x0.backward(grad.reshape(x0.shape).to(setup.device))

            reg_loss = 0.0
            if setup.reg=='origin':
                reg_loss = lam*torch.norm(z)**2
                reg_loss.backward()
            if setup.reg=='ring':
                reg_loss = lam*(torch.norm(z)-mchi)**2
                reg_loss.backward()
            #cost = databatch_loss + reg_loss

            optimizer.step()

            lam = setup.clam*lam

            data_loss = torch.sum((d-sim)**2)
            total_cost = data_loss + reg_loss

            model_loss = np.sqrt(np.mean((truemodel.flatten() - x.flatten())**2))

            if it==0: cost_min = total_cost

            if total_cost<=cost_min:
                #cost_min = total_cost
                modelmin = x[0,0,:,:] # 'decoded' model is saved only for minimum.
                #data_loss_min = data_loss.detach().numpy()
                #z_min = np.copy(z.data.numpy())
                #itdum = sg_its*it + sg

            dloss.append(data_loss.detach().cpu().numpy())
            zs.append(np.copy(z.data.cpu().numpy()))
            mloss.append(model_loss)

            if setup.fwd == 'nonlinear':
                print('it: {0}, RMSE: {1}'.format(sg_its*it+sg, np.sqrt(data_loss.detach().cpu()/ndata)))

        #if (it%step_eps) == 0:
        #    lam = lam*0.99
        scheduler.step()

    return dloss, mloss, modelmin, zs

def print_SGDsetup(setup):
    ndata = len(setup.d)
    sg_its = ndata//setup.batch_size
    its = setup.epochs*(sg_its)
    v = ("DGM: {0}, epochs: {1}, iterations: {2}, batch_size: {3}, step_eps: {4} \n"
    "lr: {5:.1e},  clr: {6:.3f}, reg: {7}, lam: {8:.1e},  clam: {9:.3f}, optimizer: {10} \n").format(setup.DGM,
         setup.epochs,its,setup.batch_size,setup.step_eps,setup.lr,setup.clr,setup.reg,setup.lam,setup.clam,
         setup.optimizer)
    print(v)
