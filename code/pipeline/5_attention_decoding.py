# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

task = "Attention"
root = "D:/EEG_wd/Machine_learning/"

# Subjects ID
names = names = [
    "31NLI", "32CVI", "34LME", "35QSY", "36LSA", "37BMA", "38MAX", "39BDA", "40MMA",
    "41BAL", "42SPE", "44SMU", "45MJA", "46SQU", "47HMA", "50JOC", "52PFA", "53SMA",
    "55MNI", "56BCL", "57NCO", "58BAN", "59DIN", "60CAN"
    ]

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

# =======================
# %% Decoding - Attention
# =======================
def run_decoding_attention(
    subject: str, 
    classifier: RandomForestClassifier
    ) -> np.ndarray:
    """
    Run a sliding decoder on the Attention task and return the 10 fold AUC scores.

    Parameters
    ----------
    subject : str
        subject reference (e.g. '31NLI')
    classifier : sklearn object
        Define the ML kernel to use.

    Return
    ------
    scores : np.ndarray
        ROC-AUC scores, time * 10 fold CV.

    References
    ----------
    ..[1] https://mne-tools.github.io/stable/auto_examples/decoding/plot_decoding_spatio_temporal_source.html

    """

    # Import behavioral and EEG data.
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    # Define features and labels
    data = attention._data[attention_df.Cond1 != "Think", :, :]
    labels = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

    # Create classification pipeline based on the provided classifier.
    clf = make_pipeline(StandardScaler(), classifier)

    # Create the sliding classifier
    time_gen = SlidingEstimator(clf, scoring="roc_auc", n_jobs=1)

    # Run a 10 fold cros validated classification
    scores = cross_val_multiscore(time_gen, data, labels, cv=8)

    # Plot single subject decoder performances
    sns.set_context("talk")
    df = pd.DataFrame(scores).melt()
    df["Time"] = (df.variable / 100) - 0.2

    # Plot scores
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Time", y="value", data=df, ci=68, color="royalblue")
    plt.axhline(y=0.5, linestyle="--", color="gray")
    plt.axvline(
        x=(scores[:, 40:70].mean(0).argmax() / 100) + 0.2, color="red", linestyle="--"
    )
    plt.axvline(x=0, color="k")
    plt.ylabel("AUC")
    plt.ylim([0.4, 1.0])
    plt.axvspan(0.25, 0.5, alpha=0.1, color="red")
    plt.annotate("Window of interest", xy=(0.1, 0.90))
    plt.savefig(root + "Results/Decoding/" + subject + "decoding.png")
    plt.clf()
    plt.close()

    return np.asarray(scores)


# =============================================================================
# %% Sliding classifier - Attention
# =============================================================================

if __name__ == "__main__":

    scores = []
    for subject in names:

        scores.append(run_decoding_attention(subject, classifier).mean(0))

    np.save(root + "Attention_decoding.npy", np.asarray(scores))
