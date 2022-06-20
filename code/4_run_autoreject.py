# Author: Nicolas Legrand (legrand@cyceron.fr)

import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject

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


def run_autoreject(subject: str, task: str):
    """Interpolate bad epochs/sensors using Autoreject.

    Parameters
    ----------
    *subject: string
        The participant reference

    Save the resulting *-epo.fif file in the '4_autoreject' directory.
    Save .png of ERP difference and heatmap plots.

    References
    ----------
    [1] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and
    Alexandre Gramfort, “Automated rejection and repair of bad trials in
    MEG/EEG.” In 6th International Workshop on Pattern Recognition in
    Neuroimaging (PRNI), 2016.

    [2] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and
    Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for
    MEG and EEG data”. NeuroImage, 159, 417-429.

    """
    # Import data
    input_path = root + task + "/4_ICA/" + subject + "-epo.fif"
    epochs = mne.read_epochs(input_path)

    # Autoreject
    ar = AutoReject(random_state=42, n_jobs=6)

    ar.fit_transform(epochs)
    epochs_clean = ar.transform(epochs)

    # Plot difference
    evoked = epochs.average()
    evoked_clean = epochs_clean.average()

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    for ax in axes:
        ax.tick_params(axis="x", which="both", bottom="off", top="off")
        ax.tick_params(axis="y", which="both", left="off", right="off")

    evoked.plot(exclude=[], axes=axes[0], ylim=[-30, 30], show=False)
    axes[0].set_title("Before autoreject")
    evoked_clean.plot(exclude=[], axes=axes[1], ylim=[-30, 30])
    axes[1].set_title("After autoreject")
    plt.tight_layout()
    plt.savefig(root + task + "/5_autoreject/" + subject + "-autoreject.png")
    plt.close()

    # Plot heatmap
    ar.get_reject_log(epochs).plot()
    plt.savefig(root + task + "/5_autoreject/" + subject + "-heatmap.png")
    plt.close()

    # Save epoch data
    out_epoch = root + task + "/5_autoreject/" + subject + "-epo.fif"
    epochs_clean.save(out_epoch)


# Loop across subjects
if __name__ == "__main__":

    for subject in names:
        for task in ["TNT", "Attention"]:
            run_autoreject(subject, task)
