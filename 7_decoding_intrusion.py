#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:38:23 2018

@author: legrand
"""
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import peakutils
import itertools
import numpy as np
import mne
import os
import pandas as pd

from mne.stats import permutation_t_test, permutation_cluster_1samp_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    accuracy_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from mne.decoding import cross_val_multiscore, GeneralizingEstimator, SlidingEstimator
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
from sklearn.model_selection import cross_val_score

root = "E:/EEG_wd/Machine_learning/"
Names = os.listdir(root + "TNT/1_raw")  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

cwd = os.getcwd()

# %% extract TNT
def data_tnt(subject):

    if subject in ["33FAM", "49STH", "54CCA"]:
        data_path = "E:/ENGRAMME/Exclus/GROUPE_2/EEG/"
        criterion_path = "E:/ENGRAMME/Exclus/GROUPE_2/COMPORTEMENT/"
    else:
        data_path = "E:/ENGRAMME/GROUPE_2/EEG/"
        criterion_path = "E:/ENGRAMME/GROUPE_2/COMPORTEMENT/"

    # Load preprocessed epochs
    in_epoch = root + "TNT/5_autoreject/" + subject + "-epo.fif"
    epochs_TNT = mne.read_epochs(in_epoch, preload=True)
    epochs_TNT.pick_types(
        emg=False, eeg=True, stim=False, eog=False, misc=False, exclude="bads"
    )

    # Load e-prime file
    eprime_df = data_path + subject + "/" + subject + "_t.txt"
    eprime = pd.read_csv(eprime_df, skiprows=1, sep="\t")
    eprime = eprime[
        [
            "Cond1",
            "Cond2",
            "Image.OnsetTime",
            "ImageFond",
            "Black.RESP",
            "ListImage.Cycle",
        ]
    ]
    eprime = eprime.drop(eprime.index[[97, 195, 293]])
    eprime["ListImage.Cycle"] = eprime["ListImage.Cycle"] - 1
    eprime.reset_index(inplace=True)

    # Droped epochs_TNT
    eprime = eprime[[not i for i in epochs_TNT.drop_log]]

    # Remove criterion
    Criterion = pd.read_csv(
        criterion_path + subject + "/TNT/criterion.txt",
        encoding="latin1",
        sep="\t",
        nrows=78,
    )
    forgotten = [
        ntpath.basename(i) for i in Criterion[" Image"][Criterion[" FinalRecall"] == 0]
    ]

    if len(forgotten):
        epochs_TNT.drop(eprime["ImageFond"].str.contains("|".join(forgotten)))
        eprime = eprime[~eprime["ImageFond"].str.contains("|".join(forgotten))]

    return epochs_TNT, eprime


# =========================================
# %% Decoding - Attention -> TNT
# =========================================


def run_decoding_attention_tnt(subject, classifier):
    """
    Run a generalized sliding decoder (GAT). Train on Attention, and
    predict probabilities on TNT.

    Parameters
    ----------
    * subject: str
        subject reference (e.g. '31NLI')
    * classifier: sklearn object
        Define the ML kernel to use.

    Return
    ------
    * scores: numpy array
        Probabilities for intrusion/no-intrusions * time * trials.
    * labels: True trial's labels
    References
    ----------
    [1]: https://mne-tools.github.io/stable/auto_examples/decoding/plot_decoding_time_generalization_conditions.html?highlight=generalization
    """

    # Load data from the attention task
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    attention.crop(
        0.2, 0.5
    )  # Only select window of interest (200ms after the intrusive image) to save memory.

    # Define training features and labels
    X_train = attention._data[attention_df.Cond1 != "Think", :, :]
    y_train = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

    # Create classifier pipeline
    clf = make_pipeline(StandardScaler(), classifier)

    # Create the GAT
    time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1)

    time_gen.fit(X_train, y_train)  # Fit the classifier to the training set.

    # Load TNT data
    tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
    tnt = mne.read_epochs(root + "TNT/6_decim/" + subject + "-epo.fif")

    # Define testing features and labels
    X_test = tnt._data[(tnt_df.Cond1 == "No-Think"), :, :]
    y_test = tnt_df["Black.RESP"][(tnt_df.Cond1 == "No-Think")] != 1

    out = time_gen.predict_proba(X_test)  # Predict probabilities

    return out, y_test


# ===========================
# %% Attention - TNT Decoding
# ===========================


def extract_decoding(subject, overwrite=True):

    """
    Run decoding the pipeline if overwrite = True, else load the .npy file.
    """
    if overwrite:

        proba, labels = run_decoding_attention_tnt(subject, classifier)

        np.save(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy", proba
        )
        np.save(
            root + "Results/Attention_TNT_decoding/" + subject + "_labels.npy", labels
        )

    else:

        proba = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy"
        )
        labels = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_labels.npy"
        )

    return proba, labels


for subject in Names:
    extract_decoding(subject, overwrite=True)

# =============================================================================
# %% Testing decoder
# =============================================================================


def testing_decoder(exclude_peak):
    """
    Find intrusions in the probabilities estimated by the decoders.
    Apply a gaussian filter and search peak. Grid search for threshold [0.5 - 1] and
    attention pattern [200 - 400ms after intrusive image apparition].

    Parameters
    ----------
    * exclude_peak: int
        Time window to exclude before intrusions (* 10ms)

    Return
    ------
    * final_df: Pandas DataFrame
        Best scores.

    * output_df: Pandas DataFrame
        All the classifiers.

    * cm_final: Pandas DataFrame
        Confusion matrix for the best classifiers.
    """

    output_df, cm_final = pd.DataFrame([]), []
    for subject in Names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba, labels = extract_decoding(subject, overwrite=False)

        high_CI = np.load(root + "Results/Shuffled_95CI/" + subject + "-high.npy")

        auc_final = 0

        for time in range(5, 30):  # 250 - 500 ms after intrusive image presentation

            for length in [1, 2, 3, 4, 5]:

                auc_time = 0  # AUC, to be maximized

                data = proba[:, time, :, 1]  # Select the probabilities of an intrusions
                ci = high_CI[:, time, :]

                total_count = []  # Number of peaks/trials
                for ii in range(len(data)):

                    cnt = 0
                    # Find all the peak > 0.5
                    indexes = peakutils.indexes(
                        data[ii, :], thres=0.5, min_dist=6, thres_abs=True
                    )

                    # Label as an intrusion if the peak > 95% CI
                    for id in indexes:

                        if (id > exclude_peak) & (
                            id < 310
                        ):  # Exclude peak < 400ms  & > 2900 ms after stim presentation

                            # Check that peak > 95 CI
                            if length == 1:
                                if data[ii, id] > (ci[ii, id]):
                                    cnt += 1
                            elif length == 2:
                                if (data[ii, id] > (ci[ii, id])) & (
                                    data[ii, id + 1] > (ci[ii, id + 1])
                                ):
                                    cnt += 1
                            elif length == 3:
                                if (
                                    (data[ii, id] > (ci[ii, id]))
                                    & (data[ii, id + 1] > (ci[ii, id + 1]))
                                    & (data[ii, id + 2] > (ci[ii, id + 2]))
                                ):
                                    cnt += 1
                            elif length == 4:
                                if (
                                    (data[ii, id] > (ci[ii, id]))
                                    & (data[ii, id + 1] > (ci[ii, id + 1]))
                                    & (data[ii, id + 2] > (ci[ii, id + 2]))
                                    & (data[ii, id + 3] > (ci[ii, id + 3]))
                                ):
                                    cnt += 1
                            elif length == 5:
                                if (
                                    (data[ii, id] > (ci[ii, id]))
                                    & (data[ii, id + 1] > (ci[ii, id + 1]))
                                    & (data[ii, id + 2] > (ci[ii, id + 2]))
                                    & (data[ii, id + 3] > (ci[ii, id + 3]))
                                    & (data[ii, id + 4] > (ci[ii, id + 4]))
                                ):
                                    cnt += 1
                    total_count.append(cnt)

                pred = (
                    np.asarray(total_count) > 0
                )  # Only prediction non-intrusion if npeaks == 0

                auc = roc_auc_score(labels, pred)  # Evaluate model accuracy

                if auc > auc_time:  # Only keep the best model.
                    auc_time = auc
                    if auc_time > auc_final:
                        auc_final = auc_time
                        cm = confusion_matrix(labels, pred)
                        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                output_df = output_df.append(
                    pd.DataFrame(
                        {
                            "Subject": subject,
                            "Time": time,
                            "Length": length,
                            "AUC": auc_time,
                        },
                        index=[0],
                    ),
                    ignore_index=True,
                )
        cm_final.append(cm)
        plt.rcParams["figure.figsize"] = [6, 6]
        sns.set_context("notebook")
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.3, vmax=0.7)
        plt.title("AUC: " + str(auc_final), size=20)
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Non-Intrusions", "Intrusion"])
        plt.yticks(tick_marks, ["Non-Intrusions", "Intrusion"], rotation=90)

        fmt = ".2f"
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                size=15,
                weight="bold",
                horizontalalignment="center",
                color="white" if cm[i, j] > 0.5 else "black",
            )
        plt.colorbar()
        plt.ylabel("True label", size=20, fontweight="bold")
        plt.xlabel("Predicted label", size=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(root + "Results/Decoding/" + subject + "tnt-decoding.png")
        plt.close()

    # Save results
    output_df.to_csv(root + "raws.txt")
    np.save(root + "Confusions.npy", np.asarray(cm_final))

    # Select best decoders
    idx = output_df.groupby(["Subject"])["AUC"].transform(max) == output_df["AUC"]
    final_df = output_df[idx]

    final_df.to_csv(root + "Classifiers.txt")

    return final_df, output_df, cm_final


# %% Run intrusion decoding
if __name__ == "__main__":

    testing_decoder(exclude_peak=40)

    # Plot averaged confusion matrix
    cm = np.load(root + "Confusions.npy").mean(0)
    final_df = pd.read_csv(root + "Classifiers.txt")
    final_df = final_df.drop_duplicates(subset="Subject", keep="first")

    plt.rcParams["figure.figsize"] = [6, 6]
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.3, vmax=0.7)
    plt.title("AUC: " + str(final_df.AUC.mean()), size=20)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Non-Intrusions", "Intrusion"])
    plt.yticks(tick_marks, ["Non-Intrusions", "Intrusion"], rotation=90)

    fmt = ".2f"
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax2 = plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            size=15,
            weight="bold",
            horizontalalignment="center",
            color="white" if cm[i, j] > 0.5 else "black",
        )

    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("True label", size=20, fontweight="bold")
    plt.xlabel("Predicted label", size=20, fontweight="bold")
    plt.savefig(cwd + "/Figures/TNT_decoding_CM.svg", dpi=300)
    plt.close()

    # AUC stripplot
    sns.set_context("talk")
    plt.rcParams["figure.figsize"] = [2, 6]
    sns.stripplot(
        data=final_df, x="AUC", orient="v", size=8, alpha=0.7, jitter=0.2, linewidth=1.5
    )
    plt.axhline(y=0.5, linestyle="--", color="r")
    plt.ylabel("AUC", size=25)
    plt.savefig(cwd + "/Figures/TNT_decoding_AUC.svg", dpi=300)
    plt.close()
