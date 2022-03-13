# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import os

import matplotlib.pyplot as plt
import mne
import pandas as pd
from autoreject import Ransac

root = "E:/EEG_wd/Machine_learning/"
names = os.listdir(root + "Attention/4_ICA/")
names = sorted(list(set([subject[:5] for subject in names])))

fname = {
    "Attention": {"eeg": "_a.fil.edf", "eprime": "_a.txt"},
    "TNT": {"eeg": "_t.fil.edf", "eprime": "_t.txt"},
}

behav_var = {
    "TNT": [
        "Cond1",
        "Cond2",
        "Image.OnsetTime",
        "Black.RESP",
        "ImageCentre",
        "ImageFond",
    ],
    "Attention": [
        "Cond1",
        "Cond2",
        "Cond3",
        "Cond4",
        "Image.OnsetTime",
        "Image.RESP",
        "Image.RT",
        "ImageCentre",
        "ImageFond",
    ],
}


def run_epochs(subject, task):
    """Run epochs.

    Transform raw data into epochs and match with experimental conditions.
    Reject bad epochs using a (high) threshold (preliminary rejection).

    Parameters
    ----------
    *subject: string
        The participant reference

    Save the resulting *-epo.fif file in the '3_epochs' directory.

    """
    if subject in ["33FAM", "49STH", "54CCA"]:
        data_path = "E:/ENGRAMME/Exclus/GROUPE_2/EEG/"
    else:
        data_path = "E:/ENGRAMME/GROUPE_2/EEG/"
    # Load filtered data
    input_path = root + task + "/2_rawfilter/" + subject + "-raw.fif"
    raw = mne.io.read_raw_fif(input_path)

    # Load e-prime df
    eprime_df = data_path + subject + "/" + subject + fname[task]["eprime"]
    eprime = pd.read_csv(eprime_df, skiprows=1, sep="\t")
    eprime = eprime[behav_var[task]]

    # Revome training rows after pause for TNT
    if task == "TNT":
        eprime = eprime.drop(eprime.index[[97, 195, 293]])
        eprime.reset_index(inplace=True)

    # Find stim presentation in raw data
    events = mne.find_events(raw, stim_channel="STI 014")

    # Compensate for delay (as measured manually with photodiod)
    events[:, 0] += int(0.015 * raw.info["sfreq"])

    # Keep only Obj Pres triggers
    events = events[events[:, 2] == 7, :]

    # Match stim presentation with conditions from eprime df
    if task == "TNT":
        for i in range(len(events)):
            if eprime["Cond1"][i] == "Think":
                if eprime["Cond2"][i] == "Emotion":
                    events[i, 2] = 1
                else:
                    events[i, 2] = 2
            elif eprime["Cond1"][i] == "No-Think":
                if eprime["Cond2"][i] == "Emotion":
                    events[i, 2] = 3
                else:
                    events[i, 2] = 4
            else:
                events[i, 2] = 5
        # Set event id
        id = {"Think/EMO": 1, "Think/NEU": 2, "No-Think/EMO": 3, "No-Think/NEU": 4}

    elif task == "Attention":

        for i in range(events.shape[0]):
            if eprime["Cond1"][i] == "Think":
                if eprime["Cond2"][i] == "Emotion":
                    events[i, 2] = 1
                else:
                    events[i, 2] = 2
            elif eprime["Cond1"][i] == "No-Think":
                if eprime["Cond2"][i] == "Emotion":
                    events[i, 2] = 3
                else:
                    events[i, 2] = 4
            else:
                events[i, 2] = 5

        # Set event id
        id = {
            "Think/EMO": 1,
            "Think/NEU": 2,
            "No-Think/EMO": 3,
            "No-Think/NEU": 4,
            "Baseline": 5,
        }

    # Epoch raw data
    tmin = -1.5
    if task == "Attention":
        tmax = 2.5
    else:
        tmax = 4
    epochs = mne.Epochs(raw, events, id, tmin, tmax, preload=True)

    # Save epochs
    epochs.save(root + task + "/3_epochs/" + subject + "-epo.fif")

    epochs.info["projs"] = list()  # remove proj

    ransac = Ransac(verbose="progressbar", n_jobs=1)
    epochs_clean = ransac.fit_transform(epochs)

    evoked = epochs.average().crop(-0.2, 3.0).apply_baseline(baseline=(None, 0))
    evoked_clean = (
        epochs_clean.average().crop(-0.2, 3.0).apply_baseline(baseline=(None, 0))
    )

    evoked.info["bads"] = ransac.bad_chs_
    evoked_clean.info["bads"] = ransac.bad_chs_

    # Evoked differences
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    for ax in axes:
        ax.tick_params(axis="x", which="both", bottom="off", top="off")
        ax.tick_params(axis="y", which="both", left="off", right="off")

    ylim = dict(grad=(-170, 200))
    evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
    axes[0].set_title("Before RANSAC")
    evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
    axes[1].set_title("After RANSAC")
    fig.tight_layout()
    plt.savefig(root + task + "/3_epochs/" + subject + "-evoked.png")
    plt.clf()
    plt.close()

    # Heatmap
    ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(ransac.bad_log.T, cmap="Reds", interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("Trials", size=15)
    ax.set_ylabel("Sensors", size=15)
    plt.setp(ax, yticks=range(7, len(ransac.picks), 10), yticklabels=ch_names)
    ax.tick_params(axis=u"both", which=u"both", length=0)
    fig.tight_layout(rect=[None, None, None, 1.1])
    ax.set_title("Bad sensors", size=25)
    plt.savefig(root + task + "/3_epochs/" + subject + "-heatmap.png")
    plt.clf()
    plt.close()

    # Save epochs
    epochs_clean.save(root + task + "/3_epochs/" + subject + "clean-epo.fif")


# Loop across subjects
if __name__ == "__main__":

    for task in ["Attention", "TNT"]:
        for subject in names:
            run_epochs(subject, task)
