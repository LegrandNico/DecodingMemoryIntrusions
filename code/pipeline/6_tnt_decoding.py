# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.forest import RandomForestClassifier
from scipy.ndimage.filters import gaussian_filter1d

import os

task = "Attention"
root = "E:/EEG_wd/Machine_learning/"
names = os.listdir(root + task + "/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

root = "E:/EEG_wd/Machine_learning/"


# =========================================
# %% Random label Decoding - Attention -> TNT
# =========================================
def shuffled_training_labels(subject, n_boot):

    """
    Run a generalized sliding decoder (GAT). Train on shuffled Attention labels.

    Parameters
    ----------
    * subject: str
        subject reference (e.g. '31NLI')

    Return
    ------
    * ci: numpy array
        Upper and lower 95% CI for a noisy classifier.
    """

    # Attention data
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    attention.crop(0.2, 0.5)  # Only select time of interest to save memory

    # TNT data
    tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
    tnt = mne.read_epochs(root + "TNT/6_decim/" + subject + "-epo.fif")

    shuffled = []

    for i in range(n_boot):

        # Classifier
        clf = make_pipeline(StandardScaler(), classifier)

        time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=6)

        X_train = attention._data[attention_df.Cond1 != "Think", :, :]
        y_train = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

        # Shuffled the trainning labels
        labels = y_train.sample(frac=1)

        # Fit the model
        time_gen.fit(X_train, labels)

        X_test = tnt._data[(tnt_df.Cond1 == "No-Think"), :, :]

        proba = time_gen.predict_proba(X_test)

        shuffled.append(proba)

    shuffled = np.asarray(shuffled)

    # 95th percentile
    ci = np.percentile(shuffled[:, :, :, :, 1], 97.5, axis=0)

    np.save(root + "Results/Shuffled_95CI/" + subject + "-high.npy", ci)


# %% Run
if __name__ == "__main__":

    for subject in names[2:]:
        shuffled_training_labels(subject, n_boot=200)
