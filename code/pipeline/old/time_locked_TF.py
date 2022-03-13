# Author: Nicolas Legrand (legrand@cyceron.fr)


import ntpath
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import peakutils
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from mne.time_frequency import tfr_morlet, tfr_multitaper
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d

task = "TNT"
root = "E:/EEG_wd/Machine_learning/"
names = os.listdir(root + task + "/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))

root = "E:/EEG_wd/Machine_learning/"

# ===========================
# %% Attention - TNT Decoding
# ===========================


def extract_decoding(subject, overwrite=True):

    """
    Run decoding the pipeline if overwrite = True, else load the .npy file.
    """
    if overwrite:

        proba, labels = run_decoding_attention_tnt(subject, classifier)

        np.save(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy", proba
        )
        np.save(
            root + "Results/Attention_TNT_decoding/" + subject + "_labels.npy", labels
        )

    else:

        proba = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy"
        )
        labels = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_labels.npy"
        )

    return proba, labels


# =============================================================================
# %% Extract time-locked
# =============================================================================


def extract_time_locked(subject, exclude_peak, info):
    """
    Extract the Time-frequency representation time-locked
    on the intrusion detections.

    Parameters
    ----------
    *subject: str
        Subject ID

    Return
    ------
    * time_locked_TFR: list
        The time-locked Time-frequency representation for intrusions.
    * time_locked_TFR_NI: list
        The time-locked Time-frequency representation for averaged non-intrusions.

    """

    # Extract TFR, average across frequencies and select only No-Think trials
    tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
    tnt_df = tnt_df[tnt_df.Cond1 == "No-Think"]
    tfr = np.load(root + "/TNT/Multitaper_NoThink/" + subject + "_No-Think_TF.npy")

    # Raw decoding + behaviors
    proba, labels = extract_decoding(subject, overwrite=False)

    # Load the selected parameters
    final_df = pd.read_csv(root + "Classifiers.txt")
    time = final_df[final_df.Subject == subject].Time.iloc[0]

    high_CI = np.load(root + "Results/Shuffled_95CI/" + subject + "-high.npy")

    data = proba[:, time, :, 1]  # Select the probabilities of an intrusion
    ci = high_CI[:, time, :]

    time_locked_TFR, time_locked_TFR_NI = (
        [],
        [],
    )  # Store the ntrials time-locked tfr and baseline

    for ii in range(len(data)):

        # Only select items where the participant reported an intrusion
        if tnt_df["Black.RESP"].iloc[ii] != 1:

            # Find all the peak > 0.5
            indexes = peakutils.indexes(
                data[ii, :], thres=0.5, min_dist=3, thres_abs=True
            )

            # Label as an intrusion if the peak > 95% CI
            for id in indexes:

                if (id > exclude_peak) & (
                    id < 240
                ):  # Exclude peak < 0.4s & > 2.5s after stim presentation

                    # Check that peak > 95 CI
                    if (
                        (data[ii, id] > (ci[ii, id]))
                        & (data[ii, id + 1] > (ci[ii, id + 1]))
                        & (data[ii, id + 2] > (ci[ii, id + 2]))
                    ):

                        # Select time locked representation for non intrusive trials in similar time points
                        baseline_tfr = tfr[
                            tnt_df["Black.RESP"] == 1, :, :, id - 20 : id + 80
                        ].mean(0)

                        # Extract the TFR for the intrusive trial
                        intrusive_tfr = tfr[ii, :, :, id - 20 : id + 80]

                        time_locked_TFR.append(intrusive_tfr)  # Time-locked
                        time_locked_TFR_NI.append(baseline_tfr)  # Time-locked

    if len(time_locked_TFR) > 5:

        # Baseline correction
        this_tfr = mne.time_frequency.EpochsTFR(
            info=info,
            data=np.asarray(time_locked_TFR),
            times=np.arange(-0.2, 0.8, 0.01),
            freqs=np.arange(3, 30),
        )
        intrusion = stats.trim_mean(
            this_tfr.apply_baseline(baseline=(-0.2, 0.0), mode="mean")._data,
            proportiontocut=0.1,
            axis=0,
        )
        np.save(
            root + "/Results/Time-locked/" + subject + "_TimeLockedIntrusion_TF.npy",
            intrusion,
        )

        this_tfr = mne.time_frequency.EpochsTFR(
            info=info,
            data=np.asarray(time_locked_TFR_NI),
            times=np.arange(-0.2, 0.8, 0.01),
            freqs=np.arange(3, 30),
        )
        baseline = stats.trim_mean(
            this_tfr.apply_baseline(baseline=(-0.2, 0.0), mode="mean")._data,
            proportiontocut=0.1,
            axis=0,
        )
        np.save(
            root + "/Results/Time-locked/" + subject + "_TimeLockedBaseline_TF.npy",
            baseline,
        )


# =============================================================================
# %% Extract global results
# =============================================================================
tnt, tnt_df = data_tnt("31NLI")

for subject in names:
    extract_time_locked(subject, exclude_peak=40, info=tnt.info)

# =============================================================================
# %%
# =============================================================================

intrusion, baseline = [], []
for subject in names:
    try:
        intrusion.append(
            np.load(
                root + "/Results/Time-locked/" + subject + "_TimeLockedIntrusion_TF.npy"
            )
        )
        baseline.append(
            np.load(
                root + "/Results/Time-locked/" + subject + "_TimeLockedBaseline_TF.npy"
            )
        )

    except:
        print(subject + " without TFR")


this_tfr = mne.time_frequency.AverageTFR(
    info=tnt.info,
    data=np.asarray(intrusion).mean(0) - np.asarray(baseline).mean(0),
    nave=len(intrusion),
    times=np.arange(-0.2, 0.6, 0.01),
    freqs=np.arange(3, 30),
)

this_tfr.plot()

# =============================================================================
# %% TF
# =============================================================================


this_tfr = mne.time_frequency.EpochsTFR(
    info=this_tfr.info,
    data=np.asarray(intrusion) - np.asarray(baseline),
    times=np.arange(-0.2, 0.6, 0.01),
    freqs=np.arange(3, 30),
)


def tfr_permutation(data, title):

    threshold = None
    n_permutations = 2000
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        data, n_permutations=n_permutations, threshold=threshold, tail=0
    )

    # Create new stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    plt.figure(figsize=(8, 4))
    plt.imshow(
        T_obs,
        cmap=plt.cm.get_cmap("RdBu_r"),
        vmin=-6,
        vmax=6,
        extent=[-0.2, 0.6, 3, 30],
        interpolation="gaussian",
        aspect="auto",
        origin="lower",
    )
    clb = plt.colorbar()
    clb.ax.set_title("t values")
    plt.contour(
        ~np.isnan(T_obs_plot),
        colors=["w"],
        extent=[-0.2, 0.6, 3, 30],
        linewidths=[2],
        corner_mask=False,
        antialiased=True,
        levels=[0.5],
    )
    plt.axvline(x=0, linestyle="--", linewidth=2, color="k")
    plt.title(title, size=20, fontweight="bold")
    plt.ylabel("Frequencies", size=15)
    plt.xlabel("Time (s)", size=15)


#    plt.savefig(title + '.svg')

pick_elec = mne.pick_channels(this_tfr.ch_names, electrodes["Frontal"])

data = this_tfr._data[:, :, :, :].mean(1)
tfr_permutation(data, "Time-locked")

# =============================================================================
# %% Topomaps
# =============================================================================


def timelocked_topomap(data, freq):

    data = data[:, :, freq, :].mean(2)

    fig, axs = plt.subplots(1, 7, figsize=(15, 5), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=0.5, wspace=0.001)

    axs = axs.ravel()

    for i, rg in enumerate(np.arange(10, 70, 10)):

        # Load data
        this_data = data[:, :, rg : rg + 10].mean(2)

        # set cluster threshold
        tail = 0  # 0.  # for two sided test
        p_thresh = 0.01 / (1 + (tail == 0))
        n_samples = len(data)
        threshold = -stats.t.ppf(p_thresh, n_samples - 1)

        # Make a triangulation between EEG channels locations to
        # use as connectivity for cluster level stat
        connectivity = mne.channels.find_ch_connectivity(
            this_tfr.average().info, "eeg"
        )[0]

        cluster_stats = permutation_cluster_1samp_test(
            this_data,
            threshold=threshold,
            verbose=True,
            connectivity=connectivity,
            out_type="indices",
            n_jobs=1,
            check_disjoint=True,
            step_down_p=0.05,
            seed=42,
        )

        T_obs, clusters, p_values, _ = cluster_stats
        good_cluster_inds = np.where(p_values < 0.05)[0]

        # Extract mask and indices of active sensors in the layout
        mask = np.zeros((T_obs.shape[0], 1), dtype=bool)
        if len(clusters):
            for clus in good_cluster_inds:
                mask[clusters[clus], :] = True

        evoked = mne.EvokedArray(
            T_obs[:, np.newaxis], this_tfr.average().info, tmin=0.0
        )

        evoked.plot_topomap(
            ch_type="eeg",
            times=0,
            scalings=1,
            time_format=None,
            cmap=plt.cm.get_cmap("RdBu_r", 12),
            vmin=-3.0,
            vmax=3,
            units="t values",
            mask=mask,
            axes=axs[i],
            size=3,
            show_names=lambda x: x[4:] + " " * 20,
            time_unit="s",
            show=False,
        )


#    plt.savefig(outpath + 'Time-locked_' + frequency + '.png', dpi=300)


data = this_tfr._data
theta = np.arange(0, 6)
alpha = np.arange(6, 10)
lowbeta = np.arange(10, 17)
highbeta = np.arange(17, 27)

timelocked_topomap(data, theta)
timelocked_topomap(data, alpha)
timelocked_topomap(data, lowbeta)
timelocked_topomap(data, highbeta)

# =============================================================================
# %% Time course
# =============================================================================

electrodes = {
    "Left-occipital": [
        "E58",
        "E59",
        "E60",
        "E64",
        "E65",
        "E66",
        "E67",
        "E69",
        "E70",
        "E71",
        "E72",
        "E74",
    ],
    "Right-occipital": [
        "E62",
        "E75",
        "E76",
        "E77",
        "E82",
        "E83",
        "E84",
        "E85",
        "E89",
        "E90",
        "E91",
        "E95",
        "E96",
    ],
    "Left-frontal": [
        "E12",
        "E18",
        "E19",
        "E20",
        "E22",
        "E23",
        "E24",
        "E26",
        "E27",
        "E28",
        "E33",
    ],
    "Right-frontal": [
        "E2",
        "E3",
        "E4",
        "E5",
        "E9",
        "E10",
        "E117",
        "E118",
        "E122",
        "E123",
        "E124",
    ],
    "Frontal": [
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E9",
        "E10",
        "E11",
        "E12",
        "E15",
        "E16",
        "E18",
        "E19",
        "E20",
        "E22",
        "E23",
        "E24",
        "E26",
        "E27",
        "E28",
        "E33",
        "E117",
        "E118",
        "E122",
        "E123",
        "E124",
    ],
    "Midline": ["E11", "E4", "E10", "E16", "E18", "E19", "E12", "E6", "E5"],
    "Occipital": [
        "E62",
        "E75",
        "E76",
        "E77",
        "E82",
        "E83",
        "E84",
        "E85",
        "E89",
        "E90",
        "E91",
        "E95",
        "E96",
        "E58",
        "E59",
        "E60",
        "E64",
        "E65",
        "E66",
        "E67",
        "E69",
        "E70",
        "E71",
        "E72",
        "E74",
    ],
}

data = this_tfr._data


def plot_time_course(data, freq, elec):
    """
    Plot the time-course of intrusive and non-intrusive TFR give a frequency and
    electrodes references.

    Parameters
    ----------
    * frequency: str
        The frequency of interest
    * elec: list
        The electrodes of interest

    Return
    ------
    * matplotlib instance
    """

    # Find electrodes references
    pick_elec = mne.pick_channels(this_tfr.ch_names, electrodes[elec])

    # Plot intrusions
    this_data = data[:, pick_elec, :, :].mean(1)
    this_data = this_data[:, freq, :].mean(1)
    df = pd.DataFrame(this_data).melt()
    df["Time"] = (df.variable / 100) - 0.2
    sns.lineplot(x="Time", y="value", data=df, ci=68, color="r")
    plt.axhline(y=0, linestyle="--", color="b")


# =============================================================================
# %%
# =============================================================================
plot_time_course(data, lowbeta, "Frontal")
