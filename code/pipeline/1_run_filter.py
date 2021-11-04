# Author: Nicolas Legrand (legrand@cyceron.fr)

import mne
import os

out = "D:/EEG_wd/Machine_learning/"
root = "D:/ENGRAMME/GROUPE_2/"
names = os.listdir(root + "EEG/")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))

# Parameter
drop = [
    "E43",
    "E49",
    "E56",
    "E63",
    "E68",
    "E73",
    "E81",
    "E88",
    "E94",
    "E99",
    "E107",
    "E113",
    "E120",
]

fname = {
    "Attention": {"eeg": "_a.fil.edf", "eprime": "_a.txt"},
    "TNT": {"eeg": "_t.fil.edf", "eprime": "_t.txt"},
}

chan_rename = {"EEG " + str(i): "E" + str(i) for i in range(1, 129)}  # Montage
chan_rename["EEG VREF"] = "Cz"

# Set channels types
mapping = {
    "E1": "eog",
    "E8": "eog",
    "E14": "eog",
    "E21": "eog",
    "E25": "eog",
    "E32": "eog",
    "E125": "eog",
    "E126": "eog",
    "E127": "eog",
    "E128": "eog",
    "E48": "emg",
    "E119": "emg",
    "E17": "misc",
    "Cz": "misc",
}


def run_filter(subject, task):
    """Filter raw data.

    Parameters
    ----------
    *subject: string
        The participant reference

    *task: string
        The file to load ('Attention' or 'TNT')

    Save the resulting *-raw.fif file in the '2_rawfilter' directory.

    """
    if subject in ["33FAM", "49STH", "54CCA"]:
        root = "D:/ENGRAMME/Exclus/GROUPE_2/EEG/"
    else:
        root = "D:/ENGRAMME/GROUPE_2/EEG/"

    # Load edf file
    subject_path = root + subject + "/" + subject + fname[task]["eeg"]
    raw = mne.io.read_raw_edf(subject_path, preload=True)

    # Rename channels
    mne.channels.rename_channels(raw.info, chan_rename)

    # Set montage
    montage = mne.channels.read_montage("GSN-HydroCel-129")
    mne.io.Raw.set_montage(raw, montage)
    raw.set_channel_types(mapping=mapping)

    # Drop channels
    raw.drop_channels(drop)

    # Save raw data
    out_raw = root + task + "/1_raw/" + subject + "-raw.fif"
    raw.save(out_raw, overwrite=True)

    # Filter
    raw.filter(
        None,
        30,
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        n_jobs=8,
        filter_length="auto",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
    )

    # Set EEG average reference
    raw.set_eeg_reference("average", projection=True)

    # Save data
    out_rawfilter = root + task + "/2_rawfilter/" + subject + "-raw.fif"
    raw.save(out_rawfilter, overwrite=True)


# Loop across subjects
if __name__ == "__main__":
    for task in ["Attention", "TNT"]:
        for subject in names:
            run_filter(subject, task)
