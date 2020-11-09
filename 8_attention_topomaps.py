# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:55:41 2019

@author: nicolas
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import trim_mean, zscore

trim = lambda x: trim_mean(x, 0.1, axis=0)

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=500, random_state=42
)

root = "E:/EEG_wd/Machine_learning/"
Names = os.listdir(root + "TNT/1_raw")  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

cwd = os.getcwd()
# =============================================================================
# %% Topomap of selected patterns
# =============================================================================

"""
Plot the averaged topomap of selected attentional patterns 
to decode intrusion.

"""

# Load the selected classifiers
final_df = pd.read_csv(root + "Classifiers.txt")
final_df = final_df.drop_duplicates(subset="Subject", keep="first")

gini = []
for subject in Names:

    time = final_df[final_df.Subject == subject].Time.iloc[0] + 40

    # Import behavioral and EEG data.
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    # Define features and labels
    data = attention._data[attention_df.Cond1 != "Think", :, time]
    labels = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

    # Create classification pipeline based on the provided classifier.
    scaler = StandardScaler()
    data = scaler.fit_transform(X=data)

    classifier.fit(data, labels)
    gini.append(classifier.feature_importances_)

attention = attention.average()

# GINI index
gini = np.asarray(gini)
for i in range(27):
    gini[i] = zscore(gini[i])

from scipy.stats import ttest_1samp

t, p = ttest_1samp(gini, axis=0, popmean=0)

T0, p_values, H0 = permutation_t_test(gini, 5000, n_jobs=1)
mask = p_values[:, np.newaxis] <= 0.05

evoked = mne.EvokedArray(T0[:, np.newaxis], attention.info, tmin=0.0)

evoked.plot_topomap(
    ch_type="eeg",
    times=0,
    scalings=1,
    time_format=None,
    cmap=plt.cm.get_cmap("RdBu_r", 12),
    vmin=-3,
    vmax=3,
    units="GINI index (z-scores)",
    mask=mask,
    size=3,
    show_names=lambda x: x[4:] + " " * 20,
    time_unit="s",
    show=False,
)
plt.savefig(cwd + "/TNT_decoding_GINI.svg", dpi=300)
