import os



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from scipy.stats import pearsonr
import itertools
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller, kpss, acf

from sklearn.preprocessing import StandardScaler


root_path_name="./dataset/"
data_path_name="Traffic.csv"

if __name__ == "__main__":
    # Load the dataset
    data_path = os.path.join(root_path_name, data_path_name)
    df = pd.read_csv(data_path)
    values = df.values[:, 1:]  # Exclude the first column (timestamp)

    sc = StandardScaler()
    scaled_values = sc.fit_transform(values)
    
    # ACF Plots
    # acf_values = acf(scaled_values[:,6], nlags=48)
    # plt.figure(figsize=(10,5))
    # plt.bar(x= range(acf_values.shape[0]), height=acf_values)
    # plt.title("ACF")
    # plt.show()

    # KPSS Test
    non_stationarity = False
    for i in range(values.shape[1]):
        pvalue = kpss(values[:,i], nlags="auto")[1]
        if pvalue < 0.05:
            non_stationarity = True
            break

    print("KPSS Test Results:")
    if non_stationarity:
        print(f"Non-Stationary for Feature {i}")
    else:
        print("Stationary")

    # ADF Test
    non_stationarity = False
    for i in range(values.shape[1]):
        pvalue = adfuller(values[:,i])[1]
        if pvalue > 0.05:
            non_stationarity = True
            break

    print("ADF Test Results:")
    if non_stationarity:
        print(f"Non-Stationary for Feature {i}")
    else:
        print("Stationary")
