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

logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)

BOOL_SPLIT_SPACE = 39
OFFSET_SPACE = 2

bs, o = None, None

def data_csv(mode):
    if mode == "only_space":
        tab = pd.read_csv("only_space.csv")
    else:
        tab = pd.read_csv("time_space.csv")
    return tab


def get_data(data, features, bool_split, offset, has_time_data=False):
    global bs
    global o
    data['target'] = data.apply(lambda row : int(row['FELONY'] + row['MISDEMEANOR'] + row['VIOLATION']), axis=1)
    bs = bool_split
    o = offset
    target = 'binary_target'
    data_new = deepcopy(data)
    data_new[target] = data_new.apply(sep_data, axis=1)
    data_new = data_new.dropna()
    data = data_new

    ###pick only the tables required#####
    months_train, months_test = None, None
    one_table = data[data[target] == 1]
    zero_table = data[data[target] == 0]
    one_indices = len(one_table)
    zero_indices = len(zero_table)

    if one_indices < zero_indices:
        zero_table = zero_table.sample(one_indices, random_state=42)
    else:
        one_table = one_table.sample(zero_indices, random_state=42)

    table = one_table.append(zero_table, ignore_index=True)

    df_X = table[features]
    df_y = table[target]

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, random_state=42)
    # Train Data
    X_np_train = normalize(np.array(X_train))
    y_np_train = np.array(y_train)

    X_nuts_train = torch.from_numpy(X_np_train).type(torch.float32)
    y_nuts_train = torch.from_numpy(y_np_train).type(torch.float32)

    # Test Data
    X_np_test = normalize(np.array(X_test))
    y_np_test = np.array(y_test)

    X_nuts_test = torch.from_numpy(X_np_test).type(torch.float32)
    y_nuts_test = torch.from_numpy(y_np_test).type(torch.float32)

    if has_time_data:
        if 'monthly_avg' in features:
            months_train = torch.from_numpy(
                np.array(
                    X_train['monthly_avg'])).type(
                torch.float32)
            months_test = torch.from_numpy(
                np.array(
                    X_test['monthly_avg'])).type(
                torch.float32)
        else:
            months_train = torch.from_numpy(
                np.array(
                    X_train['monthly_avg'])).type(
                torch.float32)
            months_test = torch.from_numpy(
                np.array(
                    X_test['monthly_avg'])).type(
                torch.float32)

    return data, X_nuts_train, y_nuts_train, months_train, X_nuts_test, y_nuts_test, months_test

def sep_data(row):
    if row['target'] >= bs + o:
        return 1
    elif row['target'] <= bs - o:
        return 0
    else:
        return float('nan')


def predict(x, y, model, guide, node1=None, node2=None,
            month=None, num_samples=100):
    posterior_pred_samples = pred(model=model, guide=guide, num_samples=num_samples)
    if not node1:
        sample_predictions = posterior_pred_samples.forward(
            x, None, torch.tensor(x.shape[1]).type(torch.long))['obs']
    else:
        if not month:
            
            sample_predictions = posterior_pred_samples.forward(
                x, None, node1, node2, torch.tensor(x.shape[1]).type(torch.long))['obs']
        else:
            sample_predictions = posterior_pred_samples.forward(
                x, None, node1, node2, torch.tensor(
                    x.shape[1]).type(
                    torch.long), month)['obs']
    mean_predictions = np.array(torch.mean(sample_predictions, axis=0))
    y_pred = [i >= 0.5 for i in mean_predictions]
    acc = accuracy_score(y_pred, y)
    return mean_predictions, y_pred, acc


def get_tnse_dict(X_nuts_train, X_nuts_test, perplexity=10):
    low_emb_test_2 = TSNE(n_components=2, perplexity=perplexity).fit_transform(X_nuts_test)
    low_emb_train_2 = TSNE(n_components=2, perplexity=perplexity).fit_transform(X_nuts_train)
    low_emb_test_3 = TSNE(n_components=3, perplexity=perplexity).fit_transform(X_nuts_test)
    low_emb_train_3 = TSNE(n_components=3, perplexity=perplexity).fit_transform(X_nuts_train)

    tnse_dict = {'2d': {'train': low_emb_train_2, 'test': low_emb_test_2},
                 '3d': {'train': low_emb_train_3, 'test': low_emb_test_3}}
    return tnse_dict


def low_dim_2d(tnse_dict, labels, n_comp=2):
    feature_vector = tnse_dict['2d']['test']
    color_map = {0: 'blue', 1: 'red'}
    plt.scatter(feature_vector[:, 0], feature_vector[:, 1], c=[
                color_map[y] for y in labels])
    plt.show()


def low_dim_3d(tnse_dict, labels, n_comp=3):
    feature_vector = tnse_dict['3d']['test']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_map = {0: 'blue', 1: 'red'}
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature_vector[:, 0], feature_vector[:, 1],
               feature_vector[:, 2], c=[color_map[y] for y in labels])
    return fig


def cm(df):
    f = plt.figure(figsize=(10, 10))
    plt.matshow(df[features_good].corr(), fignum=f.number)
    plt.xticks(
        range(
            df[features_good].shape[1]),
        df[features_good].columns,
        fontsize=14,
        rotation=45)
    plt.yticks(
        range(
            df[features_good].shape[1]),
        df[features_good].columns,
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=10)
    return plot


def get_nodes(hops=1):
    if hops == 1:
        W = pickle.load(open("weight_matrix", "rb"))
    else:
        W = pickle.load(open("weight_matrix_hops_2", "rb"))
    node1 = []
    node2 = []

    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            if W[i][j] == 1:
                node1.append(i)
                node2.append(j)
    return node1, node2


def infer(X_nuts_train, y_nuts_train, model, guide, node1=None, node2=None,
          month=None, lr=5e-2, num_samples=2000):
    svi = SVI(model,
              guide,
              optim.Adam({"lr": lr}),
              loss=JitTrace_ELBO(),
              num_samples=num_samples)
    pyro.clear_param_store()
    elbo_arr_1 = []

    if not node1:
        for i in range(10000):
            elbo = svi.step(
            X_nuts_train, y_nuts_train, torch.tensor(
                X_nuts_train.shape[1]).type(
                torch.long))
            elbo_arr_1.append(elbo)
            if i % 500 == 0:
                logging.info("Elbo loss: {}".format(elbo))
    else:
        node1 = torch.tensor(node1).type(torch.long)
        node2 = torch.tensor(node2).type(torch.long)
        if not month:
            for i in range(10000):
                elbo = svi.step(
                    X_nuts_train, y_nuts_train, node1, node2,torch.tensor(
                        X_nuts_train.shape[1]).type(
                        torch.long))
                elbo_arr_1.append(elbo)
                if i % 500 == 0:
                    logging.info("Elbo loss: {}".format(elbo))
        else:
            for i in range(10000):
                elbo = svi.step(
                    X_nuts_train, y_nuts_train, node1, node2, torch.tensor(
                        X_nuts_train.shape[1]).type(
                        torch.long), month)
                elbo_arr_1.append(elbo)
                if i % 500 == 0:
                    logging.info("Elbo loss: {}".format(elbo))

    return svi, elbo_arr_1

def get_corr(df, features):
    f = plt.figure(figsize=(10, 10))
    plt.matshow(df[features].corr(), fignum=f.number)
    plt.xticks(range(df[features].shape[1]), df[features].columns, fontsize=14, rotation=90)
    plt.yticks(range(df[features].shape[1]), df[features].columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=10);
    return plt


def heat_map(intersection_matrix):
    fig, ax = plt.subplots()

    min_val, max_val = 0, 15

    ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

    for i in range(15):
        for j in range(15):
            c = intersection_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    
def generate_base():
    df = pd.read_csv("raw_data/census_block.csv")
    df_flat_agg = pd.read_csv("raw_data/data.csv")
    ct_set = set(list(df_flat_agg['CensusTract'].unique()))
    df_census_block = pd.read_csv("data/census_block_loc.csv")
    df_census_block['CensusTract'] = df_census_block.apply(lambda row : int(row['BlockCode'])//10000, axis=1)
    df_census_block = df_census_block[df_census_block['CensusTract'].isin(ct_set)]
    df_census_block = df_census_block[['Latitude','Longitude','CensusTract','BlockCode']]
    df_census_block.to_csv('points.csv',index=False)
    lats = df_census_block['Latitude']
    longs = df_census_block['Longitude']
    unique_lats = sorted(list(set(lats)))
    lats = df_census_block['Latitude']
    longs = df_census_block['Longitude']
    unique_lats = sorted(list(set(lats)))
    unique_longs = sorted(list(set(longs)))
    pairs = set([(i,j) for (i,j) in zip(lats,longs)])
    return df_census_block, unique_lats, unique_longs, pairs
    
def generate_nyc_neighbours():
    _, unique_lats, unique_longs, pairs = generate_base()
    matr = np.zeros((len(unique_lats), len(unique_longs)))
    for lat_index, lat in enumerate(unique_lats):
        for long_index, lon in enumerate(unique_longs):
            if (lat, lon) in pairs:
                matr[lat_index][long_index] += 1

    heat_map(matr)  
    
def generate_crime_heatmap():
    df_census_block, unique_lats, unique_longs, pairs = generate_base()
    data = pd.read_csv('raw_data/all_features.csv')
    df = pd.read_csv("raw_data/all_features.csv")
    data['target'] = data.apply(lambda row : row['FELONY'] + row['MISDEMEANOR'] + row['VIOLATION'], axis=1)
    data['target'] = data['target'].astype(int)
    data['CensusTract'].astype('int32')
    df_census_block['CensusTract'].astype('int32')
    tract_lat_long = pd.DataFrame(df_census_block[['CensusTract', 'Latitude','Longitude']])
    tract_crimes = pd.DataFrame(data[['CensusTract', 'target']])
    tract_crimes = pd.DataFrame(data[['CensusTract', 'target']])
    tract_crimes.groupby(['CensusTract']).sum().to_csv('temp.csv')
    tract_crimes = pd.read_csv('temp.csv')
    lat_long_crimes = pd.merge(tract_crimes, tract_lat_long)
    lats = lat_long_crimes['Latitude']
    longs = lat_long_crimes['Longitude']
    pairs = set([(i,j) for (i,j) in zip(lats,longs)])
    crime_map = np.zeros((len(unique_lats), len(unique_longs)))
    for lat_index, lat in enumerate(unique_lats):
        for long_index, lon in enumerate(unique_longs):
            if (lat, lon) in pairs:
                crimes = lat_long_crimes[(lat_long_crimes['Latitude'] == lat) &  (lat_long_crimes['Longitude'] == lon)]['target']
                crime_map[lat_index][long_index] += crimes
    heat_map(crime_map)