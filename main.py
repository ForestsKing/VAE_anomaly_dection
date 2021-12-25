import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from model.vae import VAEAnomaly
from utils.getdata import getdata
from utils.setseed import set_seed

warnings.filterwarnings("ignore")

lr = 1e-3
epochs = 256
batch_size = 32
alpha = 0.05
rootpath = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    path = './data/SKAB/'
    filenames = os.listdir(path)
    dfs = [pd.read_csv(path + filename, sep=';', index_col='datetime', parse_dates=True) for filename in filenames]

    set_seed(0)
    predicted_outlier = []
    true_outlier = []

    for df in tqdm(dfs):
        train_x, test_x, train_y, test_y = getdata(df)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        trainset = Dataset(train_x, train_y)
        testset = Dataset(test_x, test_y)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        model = VAEAnomaly().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # train
        for e in range(epochs):
            model.train()
            trainlosses = []
            for (x, _) in trainloader:
                x = x.to(torch.float32).to(device)
                loss = model(x)
                trainlosses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # print("Epochs:", e, " || train loss: %.4f" % np.mean(trainlosses))

        # dection
        model.eval()
        with torch.no_grad():
            trues, preds = [], []
            for (x, y) in testloader:
                x = x.to(torch.float32).to(device)
                p = model.reconstructed_probability(x)

                trues.extend(y.detach().cpu().numpy())
                preds.extend(p.detach().cpu().numpy())
            trues, preds = np.array(trues), np.array(preds)

            preds = (pd.Series(preds) < alpha).astype(int).fillna(0)
            trues = trues.astype(int)
            preds = preds.values

            true_outlier.extend(trues)
            predicted_outlier.extend(preds)

    print('precision_score: %.4f' % precision_score(true_outlier, predicted_outlier))
    print('recall_score_score: %.4f' % recall_score(true_outlier, predicted_outlier))
    print('f1_score: %.4f' % f1_score(true_outlier, predicted_outlier))
