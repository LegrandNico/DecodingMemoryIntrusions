# Author: Nicolas Legrand (legrand@cyceron.fr)


import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.time_frequency import tfr_morlet
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

task = "Attention"
root = "D:/EEG_wd/Machine_learning/"
data_path = "D:/ENGRAMME/GROUPE_2/EEG/"

# Subjects ID
names = [
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


# =============================================================================
# %%
# =============================================================================
def data_attention(subject: str, eeg: bool = True):

    # Load e-prime file
    eprime_df = data_path + subject + "/" + subject + "_a.txt"
    eprime = pd.read_csv(eprime_df, skiprows=1, sep="\t")
    eprime = eprime[
        [
            "Cond1",
            "Cond2",
            "Cond3",
            "Cond4",
            "Image.OnsetTime",
            "Image.RESP",
            "Image.RT",
            "ImageCentre",
            "ImageFond",
            "ListImage.Cycle",
        ]
    ]
    eprime["ListImage.Cycle"] = eprime["ListImage.Cycle"] - 1
    epochs = None

    if eeg:

        # Load epoch from autoreject
        in_epoch = root + "Attention/5_autoreject/" + subject + "-epo.fif"
        epochs = mne.read_epochs(in_epoch, preload=True)  # Epochs
        epochs.pick_types(
            emg=False, eeg=True, stim=False, eog=False, misc=False, exclude="bads"
        )

        # Droped epochs
        eprime = eprime[[not i for i in epochs.drop_log]]
        eprime.reset_index(inplace=True, drop=True)

        # Aberrant values
        epochs.drop((eprime["Image.RT"] < 400) | (eprime["Image.RT"] > 3000))
        eprime = eprime[~((eprime["Image.RT"] < 400) | (eprime["Image.RT"] > 3000))]
        eprime.reset_index(inplace=True, drop=True)

    return epochs, eprime


# =========================================
# %% Decoding cross-frequencies - Attention
# =========================================


def attention_TF_decoding(
    subject: str, freqs: np.ndarray = np.arange(3, 30, 1), decim: int = 20
):

    # Import behavioral and EEG data.
    attention, attention_df = data_attention(subject)

    n_cycles = freqs / 2
    attention_tfr = tfr_morlet(
        attention,
        freqs,
        n_jobs=6,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
    )
    attention_tfr.crop(-0.5, 1.5)
    attention_tfr.apply_baseline(mode="percent", baseline=(None, 0))

    labels = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

    # Run a sliding decoder for each frequency band
    scores_total = []
    for this_freq in range(len(freqs)):

        data = attention_tfr._data[attention_df.Cond1 != "Think", :, this_freq, :]

        # Classifier
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                class_weight="balanced", n_estimators=50, random_state=42
            ),
        )

        time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc")

        scores = cross_val_multiscore(time_decod, data, labels, cv=8, n_jobs=8)

        scores_total.append(np.asarray(scores).mean(0))

    return np.asarray(scores_total)


# %% Run Time-frequency decoding
if __name__ == "__main__":

    total = []
    for subject in names:
        subject_score = attention_TF_decoding(subject)
        total.append(subject_score)
        np.save(
            root + "Results/Attention_TF_decoding/" + subject + ".npy", subject_score
        )

        plt.rcParams["figure.figsize"] = [10.0, 5.0]
        plt.title("Frequency decoding - " + subject, fontweight="bold")
        plt.imshow(
            subject_score,
            origin="lower",
            aspect="auto",
            vmin=0.35,
            vmax=0.65,
            cmap=plt.cm.get_cmap("RdBu_r", 20),
            interpolation="hanning",
            extent=[-0.5, 1.5, 3, 30],
        )
        plt.axvline(x=0, color="k", linewidth=3)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequencies")
        plt.colorbar()
        plt.savefig(root + "Results/Attention_TF_decoding/" + subject + ".png", dpi=300)
        plt.clf()
        plt.close()
