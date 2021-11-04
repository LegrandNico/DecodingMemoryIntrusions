# Author: Nicolas Legrand (legrand@cyceron.fr)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import peakutils
import pandas as pd
from scipy.fftpack import fftfreq

root = "E:/EEG_wd/Machine_learning/"
names = os.listdir(root + "TNT/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))


sns.set_context("talk")
plt.rcParams["figure.figsize"] = [8, 4]
plt.plot(np.arange(-0.2, 3.01, 0.01), data[i])
plt.plot(np.arange(-0.2, 3.01, 0.01), ci[i], color="#778899")
plt.fill_between(np.arange(-0.2, 3.01, 0.01), ci[i], color="#778899", alpha=0.2)
plt.axvline(x=0, linestyle="--", color="r")
plt.ylim((0, 1))
plt.xlabel("Time (s)")
plt.ylabel("Intrusion probability")
plt.tight_layout()
sns.despine()
plt.savefig("C:/Users/nicolas/Downloads/" + "non-intrusions.svg", dpi=300)
# =============================================================================
# %% Count number of mental events
# =============================================================================


def fourier(exclude_peak):
    """
    Count and plot the number of mental events.

    Parameters
    ----------
    * exclude_peak: int
        Time window to exclude before intrusions (* 10ms)

    Return
    ------
    * Matplotlib instance
    """

    # Load the selected classifiers
    final_df = pd.read_csv(root + "Classifiers.txt")
    final_df = final_df.drop_duplicates(subset="Subject", keep="first")

    total_psd = []

    for subject in names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy"
        )
        tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
        tnt_df = tnt_df[tnt_df.Cond1 == "No-Think"]

        time = final_df[final_df.Subject == subject].Time.iloc[0]

        high_CI = np.load(root + "Results/Shuffled_95CI/" + subject + "-high.npy")

        # Select the probabilities of an intrusions
        data = proba[:, time, :, 1]
        ci = high_CI[:, time, :]

        psd = []
        for i in range(len(data)):

            temp_fft = fft(data[i, :])
            temp_psd = np.abs(temp_fft) ** 2
            fr = fftfreq(len(temp_psd), 1.0 / 100)

            psd.append(temp_psd)

        temp_psd = np.asarray(psd).mean(0)
        i = fr > 0
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(fr[i], 10 * np.log10(temp_psd[i]))
        ax.set_xlabel("Frequency (1/year)")
        ax.set_ylabel("PSD (dB)")

        total_psd.append(np.mean(psd, 0))

    df = pd.DataFrame(np.asarray(total_psd)).melt()
    sns.lineplot(x="variable", y="value", data=df, ci=68, color="r")
