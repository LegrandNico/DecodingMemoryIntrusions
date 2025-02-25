# Author: Nicolas Legrand (legrand@cyceron.fr)

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA

root = "E:/EEG_wd/Machine_learning/"

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


def run_ICA(subject, task):
    """Correct epochs for EOG artifacts.

    Parameters
    ----------
    *subject: string
        The participant reference

    *task: string
        The file to open ('Attention' or 'TNT')

    Save the resulting *-epo.fif file in the '/4_ICA' directory.

    """
    input_path = root + task + "/3_epochs/" + subject + "clean-epo.fif"
    epochs = mne.read_epochs(input_path)

    # Fit ICA
    ica = ICA(n_components=0.95, method="fastica")

    picks = mne.pick_types(
        epochs.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
    )

    ica.fit(epochs, picks=picks, decim=10)

    # Uncomment to manually select ICA components
    # ica.plot_components(picks=range(12), inst=epochs)
    # eog_inds = ica.exclude

    # List of bad component to reject
    eog_inds = []

    fig, axs = plt.subplots(2, 2, figsize=(14, 6))
    axs = axs.ravel()

    for i, eo in enumerate(["E8", "E14", "E21", "E25"]):
        inds, scores = ica.find_bads_eog(epochs, ch_name=eo)
        axs[i].set_title(eo)
        axs[i].bar(np.arange(0, len(scores)), height=scores)
        axs[i].set_ylim([-0.7, 0.7])
        axs[i].axhline(y=-0.5, linestyle="--", color="r")
        axs[i].axhline(y=0.5, linestyle="--", color="r")

    #  Uncomment for manual selection
    #        for i in inds:
    #            if not i in eog_inds:
    #                eog_inds.append(i)
    plt.savefig(root + task + "/4_ICA/" + subject + "-scores.png")
    plt.clf()
    plt.close()

    # Plot components
    ica_plot = ica.plot_components(show=False)
    for i, plot in enumerate(ica_plot):
        plot
        plt.savefig(root + task + "/4_ICA/" + subject + str(i) + "-ica.png")
        plt.clf()
        plt.close()

    # Apply ICA
    ica.exclude = eog_inds
    ica.apply(epochs)

    # Save epochs
    epochs.save(root + task + "/4_ICA/" + subject + "-epo.fif")


# Loop across subjects
if __name__ == "__main__":

    for subject in names:
        for task in ["Attention", "TNT"]:
            run_ICA(subject, task)
