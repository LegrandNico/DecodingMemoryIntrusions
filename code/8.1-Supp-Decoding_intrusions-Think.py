# Author: Nicolas Legrand (legrand@cyceron.fr)

import itertools
from typing import Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import peakutils
import seaborn as sns
from mne.decoding import GeneralizingEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = "/home/nicolas/git/DecodingMemoryIntrusions/data/"

# Subjects ID
names = names = [
    "31NLI",
    "32CVI",
    "34LME",
    "35QSY",
    "36LSA",
    "37BMA",
    "38MAX",
    "39BDA",
    "40MMA",
    "41BAL",
    "42SPE",
    "44SMU",
    "45MJA",
    "46SQU",
    "47HMA",
    "50JOC",
    "52PFA",
    "53SMA",
    "55MNI",
    "56BCL",
    "57NCO",
    "58BAN",
    "59DIN",
    "60CAN",
]

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

# =========================================
# %% Decoding - Attention -> TNT
# =========================================


def run_decoding_attention_tnt(subject: str, classifier):
    """Run a generalized sliding decoder (GAT). Train on Attention, and predict
    probabilities of mental intrusion on TNT.

    Parameters
    ----------
    subject : str
        subject reference (e.g. '31NLI').
    classifier : sklearn object
        Define the ML kernel to use.
    condition : str
        The TNT condition (`"Think"` or `"No-Think"`).

    Returns
    -------
    scores : np.ndarray
        Probabilities for intrusion/no-intrusions * time * trials.
    labels : np.ndarray
        True trial's labels.

    """

    # Load data from the attention task
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    attention.crop(
        0.2, 0.5
    )  # Only select window of interest (200ms after the central image) to save memory

    # Define training features and labels
    X_train = attention._data[attention_df.Cond1 != "No-Think", :, :]
    y_train = attention_df.Cond1[attention_df.Cond1 != "No-Think"] == "Think"

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


def extract_decoding(subject: str, overwrite: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run decoding the pipeline if overwrite = True, else load the .npy file.

    Parameters
    ----------
    subject : str
        The participant ID.
    overwrite : bool
        If `True`, will overwrite the data.

    Returns
    -------
    proba : np.ndarray
        The intrusion probability.
    labels : np.ndarray
        The true labels.

    """
    if overwrite:

        proba, labels = run_decoding_attention_tnt(subject, classifier)

        np.save(
            f"{root}Results/Attention_TNT_decoding/{subject}_DecodingThink_proba.npy", proba
        )
        np.save(
            f"{root}Results/Attention_TNT_decoding/{subject}_DecodingThink_labels.npy", labels
        )

    else:

        proba = np.load(
            f"{root}Results/Attention_TNT_decoding/{subject}_DecodingThink_proba.npy"
        )
        labels = np.load(
            f"{root}Results/Attention_TNT_decoding/{subject}_DecodingThink_labels.npy"
        )

    return proba, labels


# =============================================================================
# %% Testing decoder
# =============================================================================


def testing_decoder(
    exclude_peak: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find intrusions in the probabilities estimated by the decoders.

    Parameters
    ----------
    exclude_peak : int
        Time window to exclude before intrusions (* 10ms).

    Returns
    -------
    final_df : pd.DataFrame
        Best scores.
    output_df : pd.DataFrame
        All the classifiers.
    cm_final : pd.DataFrame
        Confusion matrix for the best classifiers.

    """

    output_df, cm_final = pd.DataFrame([]), []
    for subject in names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba, labels = extract_decoding(
            subject, overwrite=True
            )

        high_CI = np.load(
            f"{root}Results/Shuffled_95CI/DecodingThink/{subject}-high.npy"
            )

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
                            id < 317
                        ):  # Exclude peak < 200ms  & > 2970 ms after stim presentation

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
                
                output_df = pd.concat(
                    [output_df,
                    pd.DataFrame(
                        {
                            "Subject": subject,
                            "Condition": "DecodingThink",
                            "Time": time,
                            "Length": length,
                            "AUC": auc_time,
                            "Percent_intrusion": None,
                        },
                        index=[0],
                    )],
                    ignore_index=True,
                )

                # Save decoding plot for this participant
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
                plt.savefig(f"{root}Results/Decoding/{subject}_DecodingThink_tnt-decoding.png")
                plt.close()

    # Save results
    output_df.to_csv(root + "raws_DecodingThink.txt")
    np.save(root + "Confusions_DecodingThink.npy", np.asarray(cm_final))

    # Select best decoders
    idx = output_df.groupby(["Subject", "Condition"])["AUC"].transform(max) == output_df["AUC"]
    final_df = output_df[idx]

    # Add the corresponding Think condition given the best decoder for intrusions
    for subject in names:
        time = final_df[(final_df.Subject == subject)].Time.values[0]
        length = final_df[(final_df.Subject == subject)].Length.values[0]
        final_df = final_df.append([
            output_df[
                (output_df.Condition == "Think") & (output_df.Subject == subject) & (output_df.Time == time) & (output_df.Length == length)
                ]
        ])

    final_df.to_csv(root + "Classifiers_DecodingThink.txt")

    return final_df, output_df, cm_final


# %% Run intrusion decoding
if __name__ == "__main__":

    testing_decoder(exclude_peak=40)

    # Plot averaged confusion matrix
    cm = np.load(root + "Confusions_DecodingThink.npy").mean(0)
    final_df = pd.read_csv(root + "Classifiers_DecodingThink.txt")
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
    plt.savefig(f"{root}Results/Decoding/TNT_decoding_CM_DecodingThink.svg", dpi=300)
    plt.close()

    # AUC stripplot
    sns.set_context("talk")
    plt.rcParams["figure.figsize"] = [2, 6]
    sns.stripplot(
        data=final_df, y="AUC", orient="v", size=8, alpha=0.7, jitter=0.2, linewidth=1.5
    )
    plt.axhline(y=0.5, linestyle="--", color="r")
    plt.ylabel("AUC", size=25)
    plt.savefig(f"{root}Results/Decoding/TNT_decoding_AUC_DecodingThink.svg", dpi=300)
    plt.close()
