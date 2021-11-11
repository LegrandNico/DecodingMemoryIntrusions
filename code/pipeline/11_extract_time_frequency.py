# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import pandas as pd
import numpy as np
import os
import ntpath
import mne
from mne.time_frequency import tfr_multitaper, tfr_morlet

task = "TNT"
root = "D:/EEG_wd/Machine_learning/"
names = os.listdir(root + task + "/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))

root = "D:/EEG_wd/Machine_learning/"

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


# =============================================================================
# %% Extract TF
# =============================================================================


def extract_frequencies(subject, freqs, decim):
    """Filter No-Think epochs using multitaper.

    Input
    -----
    * subject: str
        Subject reference.

    freqs: array like
        Frequency range to extract.

    decim: int
        Decimation parameter

    Output
    ------

    Save -tfr.h5 in the '/All_frequencies_multitaper' directory

    """
    n_cycles = freqs / 2

    # Import TNT data
    tnt, tnt_df = data_tnt(subject)

    epochs = tnt[tnt_df.Cond1 == "No-Think"]

    this_tfr = tfr_morlet(
        epochs,
        freqs,
        n_cycles=n_cycles,
        n_jobs=6,
        average=False,
        decim=decim,
        return_itc=False,
    )

    this_tfr = this_tfr.crop(-0.2, 3.0)
    this_tfr = this_tfr.apply_baseline(mode="mean", baseline=(-0.2, 0))

    np.save(
        root + "/TNT/Multitaper_NoThink/" + subject + "_No-Think_TF.npy", this_tfr._data
    )


# =============================================================================
# %%
# =============================================================================
total = []
for subject in names:
    extract_frequencies(subject, freqs=np.arange(3, 30, 1), decim=10)
