from __future__ import absolute_import, division, print_function

import warnings
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO, TracePredictive
from pyro.infer.predictive import Predictive as pred
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.util import diagnostics

import pyro.optim as optim
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve
from pyro.ops.stats import waic
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from copy import deepcopy
from imblearn.under_sampling import RandomUnderSampler
import pickle
from mpl_toolkits.mplot3d import Axes3D


pyro.set_rng_seed(1)
# assert pyro.__version__.startswith('0.4.1')
warnings.filterwarnings('ignore')

# %matplotlib inline
logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)


si_alpha = 0.5
si_beta = 0.5

usi_alpha = 0.3
usi_beta = 0.5


def model_baseline(data_x, data_y, D):
    beta = []
    for i in range(D + 1):
        beta.append(
            pyro.sample(
                "beta" + str(i),
                dist.Normal(
                    torch.tensor(0.),
                    torch.tensor(10.))))

    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i + 1] * data_x[:, i]

    p = torch.tensor(1.) / (torch.tensor(1.) + torch.exp(-logit))

    with pyro.plate("data", len(data_x)):
        y = pyro.sample("obs", dist.Bernoulli(p), obs=data_y)

    return p


def guide_baseline(data_x, data_y, D):

    mu_i = []
    sigma_i = []
    for i in range(D + 1):
        mu_i.append(
            pyro.param(
                'mu_i' + str(i),
                torch.tensor(0.),
                constraint=constraints.real))
        sigma_i.append(
            pyro.param(
                'sigma_i' + str(i),
                torch.tensor(10.),
                constraint=constraints.positive))

    beta = []
    for i in range(D + 1):
        beta.append(
            pyro.sample(
                "beta" + str(i),
                dist.Normal(
                    mu_i[i],
                    sigma_i[i])))

    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i + 1] * data_x[:, i]

    p = 1. / (1 + torch.exp(-logit))


def model_2(data_x, data_y, D):
    beta = []
    for i in range(D + 1):
        beta.append(pyro.sample("beta" + str(i), dist.Normal(0., 1.)))
    sd_u = pyro.sample("sd_u", dist.Gamma(usi_alpha, usi_beta))
    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i + 1] * data_x[:, i]
    sigma2_ui = 1 / (sd_u**2)
    u_i = pyro.sample("u_i", dist.Normal(0., sigma2_ui))
    logit += u_i
    p = 1. / (1 + torch.exp(-logit))
    with pyro.plate("data", len(data_x)):
        y = pyro.sample("obs", dist.Bernoulli(p), obs=data_y)
    return p


def guide_2(data_x, data_y, D):
    mu_i = []
    sigma_i = []
    for i in range(D + 1):
        mu_i.append(
            pyro.param(
                'mu_i' + str(i),
                torch.tensor(0.),
                constraint=constraints.real))
        sigma_i.append(
            pyro.param(
                'sigma_i' + str(i),
                torch.tensor(1.),
                constraint=constraints.positive))
    beta = []
    for i in range(D + 1):
        #         print(mu_i[i],sigma_i[i])
        beta.append(
            pyro.sample(
                "beta" + str(i),
                dist.Normal(
                    mu_i[i],
                    sigma_i[i])))
    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i + 1] * data_x[:, i]
    alpha_gamma = pyro.param(
        'alpha_gamma',
        torch.tensor(usi_alpha).type(
            torch.float32),
        constraint=constraints.positive)
    beta_gamma = pyro.param(
        'beta_gamma',
        torch.tensor(usi_beta).type(
            torch.float32),
        constraint=constraints.positive)
    sd_u = min(pyro.sample("sd_u", dist.Gamma(alpha_gamma, beta_gamma)), 500)
    sigma2_ui = 1 / ((sd_u + 1)**2)
#     print('sigma : ',sigma2_ui)
    u_i = pyro.sample("u_i", dist.Normal(0., sigma2_ui))

    logit += u_i

    p = 1. / (1 + torch.exp(-logit))

    return p


def model_icar(data_x, data_y, node1, node2, D):
    beta = []
    for i in range(D + 1):
        beta.append(pyro.sample("beta" + str(i), dist.Normal(0., 1.0)))

    logit = beta[0]
    for i in range(data_x.shape[1]):
        logit = logit + beta[i + 1] * data_x[:, i]

    # unstructured random noise
    sd_u = pyro.sample("sd_u", dist.Gamma(usi_alpha, usi_beta))
    sigma2_ui = 1 / (sd_u**2)
    u_i = pyro.sample("u_i", dist.Normal(0, sigma2_ui))
    logit += u_i

    # structured random noise
    sd_s = pyro.sample("sd_s", dist.Gamma(si_alpha, si_beta))
    sigma2_si = 1 / (sd_s**2)

    sum_p = 0
    phi = []
    for i in range(2078):
        p = pyro.sample("phi" + str(i), dist.Normal(0., 1.))
        phi.append(p)
        sum_p += p
    phi.append(-sum_p)

    phi = torch.from_numpy(np.array(phi, dtype='float32')).type(torch.float32)
    diff = phi[node1] - phi[node2]
    phi_joint = torch.exp(-0.5 * torch.dot(diff, diff))

    logit = phi_joint * sigma2_si

    p = 1. / (1 + torch.exp(-logit))

    with pyro.plate("data", len(data_x)):
        y = pyro.sample("obs", dist.Bernoulli(p), obs=data_y)

    return p


def guide_icar(data_x, data_y, node1, node2, D):
    mu_i = []
    sigma_i = []
    for i in range(D + 1):
        mu_i.append(
            pyro.param(
                'mu_i' + str(i),
                torch.tensor(0.),
                constraint=constraints.real))
        sigma_i.append(
            pyro.param(
                'sigma_i' + str(i),
                torch.tensor(1.),
                constraint=constraints.positive))

    beta = []
    for i in range(D + 1):
        beta.append(
            pyro.sample(
                "beta" + str(i),
                dist.Normal(
                    mu_i[i],
                    sigma_i[i])))

    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i + 1] * data_x[:, i]

    alpha_gamma = pyro.param(
        'alpha_gamma',
        torch.tensor(usi_alpha).type(
            torch.float32),
        constraint=constraints.positive)
    beta_gamma = pyro.param(
        'beta_gamma',
        torch.tensor(usi_beta).type(
            torch.float32),
        constraint=constraints.positive)

    sd_u = pyro.sample("sd_u", dist.Gamma(alpha_gamma, beta_gamma))
    sigma2_ui = 1 / ((sd_u + 1)**2)
    u_i = pyro.sample("u_i", dist.Normal(0., sigma2_ui))

    alpha_si = pyro.param(
        'alpha_si',
        torch.tensor(si_alpha).type(
            torch.float32),
        constraint=constraints.positive)
    beta_si = pyro.param(
        'beta_si',
        torch.tensor(si_beta).type(
            torch.float32),
        constraint=constraints.positive)

    sd_s = pyro.sample("sd_s", dist.Gamma(alpha_si, beta_si))
    sigma2_si = 1 / (sd_s**2)

    phi = []
    sum_p = 0
    for i in range(2078):
        p = pyro.sample("phi" + str(i), dist.Normal(0., 1.))
        phi.append(p)
        sum_p += p
    phi.append(-sum_p)

    phi = torch.from_numpy(np.array(phi, dtype='float32')).type(torch.float32)
    diff = phi[node1] - phi[node2]

    phi_joint = torch.exp(-0.5 * torch.dot(diff, diff))

    logit = phi_joint * sigma2_si


#     logit += u_i

    p = 1. / (1 + torch.exp(-logit))

    return p



def model_spatio_temporal_linear(data_x, data_y, node1, node2,D, months):
    beta = []
    for i in range(D+1):
        beta.append(pyro.sample("beta"+str(i), dist.Normal(0., 1.0)))
    
    logit = beta[0]
    for i in range(data_x.shape[1]):
        logit = logit + beta[i+1]*data_x[:,i]
    
    #unstructured random noise
    sd_u = pyro.sample("sd_u",dist.Gamma(0.5,0.5))
    sigma2_ui = 1/(sd_u**2) 
    u_i = pyro.sample("u_i",dist.Normal(0, sigma2_ui))
    
    #structured random noise
    sd_s = pyro.sample("sd_s",dist.Gamma(.5,.5))
    sigma2_si = 1/(sd_s**2) 
    
    sum_p = 0
    phi = []
    for i in range(2078):
        p = pyro.sample("phi"+str(i), dist.Normal(0.,1.))
        phi.append(p)
        sum_p += p
    phi.append(-sum_p)

    phi = torch.from_numpy(np.array(phi, dtype='float32')).type(torch.float32)
    diff = phi[node1] - phi[node2]
    phi_joint = torch.exp(-0.5 * torch.dot(diff,diff)) 
    
    logit += phi_joint*sigma2_si
    
    # Spatio-temporal
    sd_d = pyro.sample("sd_d",dist.Gamma(.5,.5))
    sigma2_delta_i = 1/(sd_d**2)
    
    sum_delta = 0
    delta = []
    for i in range(2078):
        de = pyro.sample("delta"+str(i), dist.Normal(0.,1.))
        delta.append(de)
        sum_delta += de
    delta.append(-sum_delta)

    delta = torch.from_numpy(np.array(delta, dtype='float32')).type(torch.float32)
    diff_delta = delta[node1] - delta[node2]
    delta_joint = torch.exp(-0.5 * torch.dot(diff_delta,diff_delta)) 
    
    logit += delta_joint*sigma2_delta_i*months

        
    p = 1. / (1 + torch.exp(-logit)) 


    with pyro.plate("data", len(data_x)):
        y = pyro.sample("obs",dist.Bernoulli(p), obs=data_y)
        
    return p
        
def guide_spatio_temporal_linear(data_x, data_y, node1, node2,D, months):
    mu_i = []
    sigma_i = []
    for i in range(D+1):
        mu_i.append(pyro.param('mu_i'+str(i),torch.tensor(0.),constraint = constraints.real))
        sigma_i.append(pyro.param('sigma_i'+str(i),torch.tensor(1.), constraint = constraints.positive))
    
    beta = []
    for i in range(D+1):
        beta.append(pyro.sample("beta"+str(i), dist.Normal(mu_i[i], sigma_i[i])))     

    logit = beta[0]
    for i in range(D):
        logit = logit + beta[i+1]*data_x[:,i]
    
    alpha_gamma = pyro.param('alpha_gamma',torch.tensor(0.5).type(torch.float32),constraint = constraints.positive)
    beta_gamma = pyro.param('beta_gamma',torch.tensor(0.5).type(torch.float32),constraint = constraints.positive)
      
    sd_u = pyro.sample("sd_u",dist.Gamma(alpha_gamma, beta_gamma))
    sigma2_ui = 1/((sd_u+1)**2) 
    u_i = pyro.sample("u_i",dist.Normal(0., sigma2_ui))
    
    
    alpha_si = pyro.param('alpha_si',torch.tensor(.5).type(torch.float32),constraint = constraints.positive)
    beta_si = pyro.param('beta_si',torch.tensor(.5).type(torch.float32),constraint = constraints.positive)
    
    alpha_sd = pyro.param('alpha_sd',torch.tensor(.5).type(torch.float32),constraint = constraints.positive)
    beta_sd = pyro.param('beta_sd',torch.tensor(.5).type(torch.float32),constraint = constraints.positive)
      
    sd_s = pyro.sample("sd_s",dist.Gamma(alpha_si,beta_si))
    sigma2_si = 1/(sd_s**2) 
    
    phi = []
    sum_p = 0
    for i in range(2078):
        p = pyro.sample("phi"+str(i), dist.Normal(0.,1.))
        phi.append(p)
        sum_p += p
    phi.append(-sum_p)
    phi = torch.from_numpy(np.array(phi, dtype='float32')).type(torch.float32)
    diff = phi[node1] - phi[node2]  
    phi_joint = torch.exp(-0.5 * torch.dot(diff,diff)) 
    
    logit += phi_joint*sigma2_si
    
    sd_d = pyro.sample("sd_d",dist.Gamma(alpha_sd,beta_sd))
    sigma2_delta_i = 1/(sd_d**2)
    
    sum_delta = 0
    delta = []
    for i in range(2078):
        de = pyro.sample("delta"+str(i), dist.Normal(0.,1.))
        delta.append(de)
        sum_delta += de
    delta.append(-sum_delta)

    delta = torch.from_numpy(np.array(delta, dtype='float32')).type(torch.float32)
    diff_delta = delta[node1] - delta[node2]
    delta_joint = torch.exp(-0.5 * torch.dot(diff_delta,diff_delta)) 
    logit += delta_joint*sigma2_delta_i*months
    
    logit += u_i
    
    p = 1. / (1 + torch.exp(-logit))
    
    return p

