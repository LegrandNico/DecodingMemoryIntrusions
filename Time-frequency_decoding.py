# Author: Nicolas Legrand (legrand@cyceron.fr)


import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mne.time_frequency import tfr_multitaper, tfr_morlet
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.forest import RandomForestClassifier


task = "Attention"
root = "E:/EEG_wd/Machine_learning/"
Names = os.listdir(root + task + "/1_raw")  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

root = "E:/EEG_wd/Machine_learning/"

# =============================================================================
# %%
# =============================================================================


def data_attention(subject, eeg=True):

    if subject in ["33FAM", "49STH", "54CCA"]:
        data_path = "E:/ENGRAMME/Exclus/GROUPE_2/EEG/"
    else:
        data_path = "E:/ENGRAMME/GROUPE_2/EEG/"

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


def attention_TF_decoding(subject, freqs=np.arange(3, 30, 1), decim=20):

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
    for subject in Names[12:]:
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

# Load total
total = []
for subject in Names:
    score = np.load(root + "Results/Attention_TF_decoding/" + subject + ".npy")
    total.append(score)

# Plot average results
threshold = None
n_permutations = 5000
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    np.asarray(total) - 0.5, n_permutations=n_permutations, threshold=threshold, tail=0
)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

plt.rcParams["figure.figsize"] = [10.0, 5.0]
max_decod = np.unravel_index(np.asarray(total).mean((0)).argmax(), [27, 101])[1]
plt.title("Time-frequency decoding", fontweight="bold", size=25)
plt.imshow(
    np.asarray(total).mean(0),
    origin="lower",
    aspect="auto",
    vmin=0.35,
    vmax=0.65,
    cmap=plt.cm.get_cmap("RdBu_r"),
    interpolation="hanning",
    extent=[-0.5, 1.5, 3, 30],
)
plt.colorbar()
plt.contour(
    ~np.isnan(T_obs_plot),
    colors=["k"],
    extent=[-0.5, 1.5, 3, 30],
    linewidths=[2],
    corner_mask=False,
    antialiased=True,
    levels=[0.5],
)
plt.axvline(x=0, color="k", linewidth=3)
plt.axvline(x=(max_decod / 50) - 0.5, linestyle="--", linewidth=3)
plt.xlabel("Time (s)", size=20)
plt.ylabel("Frequencies (Hz)", size=20)
plt.savefig("frequency_decoding.svg", dpi=300)


plt.rcParams["figure.figsize"] = [5.0, 2.0]
df = pd.DataFrame(np.asarray(total)[:, :, max_decod]).melt()
sns.lineplot(x="variable", y="value", markers=True, data=df, ci=68, color="r")
plt.axhline(y=0.5, linestyle="--")
plt.savefig("frequency_decoding2.svg", dpi=300)
