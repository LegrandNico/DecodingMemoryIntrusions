# Author: Nicolas Legrand (legrand@cyceron.fr)

import ntpath

import mne
import pandas as pd

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

# Files name
fname = {
    "TNT": {"eeg": "_t.fil.edf", "eprime": "_t.txt"},
    "Attention": {"eeg": "_a.fil.edf", "eprime": "_a.txt"},
}


def data_attention(subject, eeg=True):

    data_path = "E:/ENGRAMME/GROUPE_2/EEG/"

    # Load e-prime file
    eprime_df = data_path + subject + "/" + subject + fname["Attention"]["eprime"]
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


# %% extract TNT
def data_tnt(subject):

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
    criterion = pd.read_csv(
        criterion_path + subject + "/TNT/criterion.txt",
        encoding="latin1",
        sep="\t",
        nrows=78,
    )
    forgotten = [
        ntpath.basename(i) for i in criterion[" Image"][criterion[" FinalRecall"] == 0]
    ]

    if len(forgotten):
        epochs_TNT.drop(eprime["ImageFond"].str.contains("|".join(forgotten)))
        eprime = eprime[~eprime["ImageFond"].str.contains("|".join(forgotten))]

    return epochs_TNT, eprime


# %% Extract Raws data


def extract_raws(subject, decim):
    """Extract raw data from proprocessed epochs. Crop, decimate and apply baseline to
    save memory.

    Parameters
    ----------
    subject: str
        subject reference (e.g. '31NLI')
    decim: int
        Decimate parameter.

    Notes
    -----
        This function Will save the processed .fif data in /Raws folder and the .txt
        behavioral data in the `/Behavior` folder.

    """
    # Extract data from the Attention task
    attention, attention_df = data_attention(subject)

    attention.decimate(decim)
    attention.crop(-0.2, 1.5)
    attention.apply_baseline(baseline=(None, 0))
    attention.save(root + "Attention/6_decim/" + subject + "-epo.fif")
    attention_df.to_csv(root + "Attention/Behavior/" + subject + ".txt")

    # Extract data from the TNT task
    tnt, tnt_df = data_tnt(subject)
    tnt.decimate(decim)
    tnt.crop(-0.2, 3.0)
    tnt.apply_baseline(baseline=(None, 0))
    tnt.save(root + "TNT/6_decim/" + subject + "-epo.fif")
    tnt_df.to_csv(root + "TNT/Behavior/" + subject + ".txt")


# =============================================================================
# %%
# =============================================================================

if __name__ == "__main__":
    for subject in names:  #
        decim = 10
        extract_raws(subject, decim)
