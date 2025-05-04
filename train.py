

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utillc import *
import os, glob, sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import opt_transport
import sklearn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

do_plot = False

class OTDataset(Dataset):
    def __init__(self) :
        files = glob.glob("/mnt/hd3/data/generated/ot/ot_*.pckl")
        EKON(len(files))
        self.dataset = [ e for f in files for e in pickle.load(open(f, 'rb'))]
        self.dataset =  self.dataset[0:100_000]
        EKON(len(self.dataset))

    def plot_sample(self) :
        for e in self.dataset :
            cst, mx, mm, ss = e
            (cst1, mx, mm, ss), (a, b, M, G0, cst2) = opt_transport.ev_(mx, mm, ss)
            plt.plot([i for i,_ in enumerate(a)], a, b)
            EKON(cst1, cst)
            plt.title(str(cst))
            plt.show()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        e = self.dataset[idx]
        cost, mx, mm, ss = e
        #EKOX(np.asarray(mx).shape)
        x = np.hstack((np.asarray(mx)[0:3:2], mm, ss))
        sample = {'y' : cost, 'X' : x}
        return sample

original_dataset = OTDataset()
if do_plot :
    original_dataset.plot_sample()

scaler = sklearn.preprocessing.StandardScaler()
dX = np.asarray([ e['X'] for e in original_dataset])
#px = pd.DataFrame(dX)
#EKON(px.describe())

#dX = scaler.fit_transform(dX)
pX = pd.DataFrame(dX)
EKON(pX.describe())

dy = np.asarray([ e['y'] for e in original_dataset])[:,None]
#dy = scaler.fit_transform(dy)
py = pd.DataFrame(dy)
EKON(py.describe())

EKON(dX.shape, dy.shape)

if do_plot :
    pdf = pd.DataFrame(np.hstack((dX, dy)))
    pdf = pdf.set_axis(["mix%02d" % e for e in range(4)] +
                       ["moy%02d" % e for e in range(4)] +
                       ["std%02d" % e for e in range(4)] + [ 'cost' ],
                       axis=1)
    sns.pairplot(pdf)
    plt.show()



full_dataset = torch.utils.data.TensorDataset(torch.tensor(dX), torch.tensor(dy))
_, d_in = dX.shape
EKOX(d_in)

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.7, 0.3])

bs = 64*8

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

def array_from_dl(dl) :
    _dx = np.asarray([r for e in dl for r in e[0].numpy() ])
    _dy = np.asarray([r for e in dl for r in e[1].numpy() ])
    return _dx, _dy

tX, ty = array_from_dl(training_loader)
vX, vy = array_from_dl(validation_loader)

regressor = DecisionTreeRegressor(max_depth=20)
EKON(mean_squared_error(regressor.fit(tX, ty).predict(tX), ty))
EKON(mean_squared_error(regressor.fit(tX, ty).predict(vX), vy))

regressor = SVR(C=1.0, epsilon=0.2)
#EKON(mean_squared_error(regressor.fit(tX, ty).predict(vX), vy))


width, depth = 20, 50
layers_f = lambda : [ nn.Dropout(0.4), nn.Linear(width, width), nn.ReLU()]
layers = ([ nn.Linear(d_in, width), nn.ReLU() ] +
          layers_f()  * depth +
          [ nn.Linear(width, 1),
#            nn.Sigmoid()
           ])
#EKON(layers)
model = nn.Sequential(*layers)

todev = lambda m : m.cuda()

model = todev(model)

loss_fn = nn.MSELoss()
epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
sched = OneCycleLR(optimizer, max_lr=0.00001,
                   steps_per_epoch=bs,
                   epochs=epochs)

for epoch in range(epochs):
    EKOX(epoch)
    def fe_(e) :
        y, X = todev(e[1].float()), todev(e[0].float())
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        return loss
    
    for e in training_loader :
        loss = fe_(e)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sched.step()
        
    fe = lambda x : fe_(x).item()
        
    with torch.no_grad():
        mse_val = np.mean(list(map(fe, validation_loader)))
        mse_train = np.mean(list(map(fe, training_loader)))
        EKON(mse_val, mse_train)
        
        


    
