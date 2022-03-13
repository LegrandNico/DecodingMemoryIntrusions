# Author: Nicolas Legrand (legrand@cyceron.fr)

# Extract TFR, average across frequencies and select only No-Think trials
tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
tnt_df = tnt_df[tnt_df.Cond1 == "No-Think"]
tfr = np.load(root + "/TNT/Multitaper_NoThink/" + subject + "_No-Think_TF.npy")

this_tfr = mne.time_frequency.EpochsTFR(
    info=info,
    data=np.asarray(tfr),
    times=np.arange(-0.2, 3.01, 0.01),
    freqs=np.arange(3, 30),
)
this_tfr = this_tfr.apply_baseline(baseline=(-0.2, 0.0), mode="percent")._data

l = []
for i in np.arange(1, 9, 1):

    l.append(tfr[tnt_df["ListImage.Cycle"] == i, :, 10:, 40:].mean())


# =============================================================================
# %% Count EEG intrusions
# =============================================================================


def count_intrusions(exclude_peak):
    """
    Count and plot the number of intrusions.

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

    intrusion_df = pd.DataFrame([])

    for subject in names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy"
        )
        labels = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_labels.npy"
        )
        tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
        tnt_df = tnt_df[tnt_df.Cond1 == "No-Think"]

        time = final_df[final_df.Subject == subject].Time.iloc[0]
        length = final_df[final_df.Subject == subject].Length.iloc[0]

        high_CI = np.load(root + "Results/Shuffled_95CI/" + subject + "-high.npy")

        # Select the probabilities of an intrusions
        data = proba[:, time, :, 1]
        ci = high_CI[:, time, :]

        total_count = []  # Number of peaks/trials
        for ii in range(len(data)):

            cnt = 0
            # Find all the peak > 0.5
            indexes = peakutils.indexes(
                data[ii, :], thres=0.5, min_dist=6, thres_abs=True
            )

            # Label as an intrusion if the peak > 95% CI
            for id in indexes:

                if (id > exclude_peak) & (
                    id < 310
                ):  # Exclude peak < 400ms after stim presentation

                    # Check that peak > 95 CI
                    if length == 1:
                        if data[ii, id] > (ci[ii, id]):
                            cnt += 1
                    elif length == 2:
                        if (data[ii, id] > (ci[ii, id])) & (
                            data[ii, id + 1] > (ci[ii, id + 1])
                        ):
                            cnt += 1
                    elif length == 3:
                        if (
                            (data[ii, id] > (ci[ii, id]))
                            & (data[ii, id + 1] > (ci[ii, id + 1]))
                            & (data[ii, id + 2] > (ci[ii, id + 2]))
                        ):
                            cnt += 1
                    elif length == 4:
                        if (
                            (data[ii, id] > (ci[ii, id]))
                            & (data[ii, id + 1] > (ci[ii, id + 1]))
                            & (data[ii, id + 2] > (ci[ii, id + 2]))
                            & (data[ii, id + 3] > (ci[ii, id + 3]))
                        ):
                            cnt += 1
                    elif length == 5:
                        if (
                            (data[ii, id] > (ci[ii, id]))
                            & (data[ii, id + 1] > (ci[ii, id + 1]))
                            & (data[ii, id + 2] > (ci[ii, id + 2]))
                            & (data[ii, id + 3] > (ci[ii, id + 3]))
                            & (data[ii, id + 4] > (ci[ii, id + 4]))
                        ):
                            cnt += 1
            total_count.append(cnt)

        pred = (
            np.asarray(total_count) > 0
        )  # Only prediction non-intrusion if npeaks == 0

        # Save predicted intrusion in df
        tnt_df["EEG_intrusions"] = np.asarray(pred)

        for block in range(1, 9):

            for emotion in ["Emotion", "Neutral"]:

                intrusions = (
                    tnt_df[tnt_df["ListImage.Cycle"] == block]["Black.RESP"] != 1
                )
                eeg_intrusions = tnt_df[tnt_df["ListImage.Cycle"] == block][
                    "EEG_intrusions"
                ]

                intrusion_df = intrusion_df.append(
                    pd.DataFrame(
                        {
                            "Subject": subject,
                            "Block": block,
                            "Reported": intrusions.sum() / len(intrusions),
                            "Decoded": eeg_intrusions.sum() / len(eeg_intrusions),
                        },
                        index=[0],
                    ),
                    ignore_index=True,
                )

        # Plot averaged subjective vs decoding intrusions
        df = pd.melt(
            intrusion_df[intrusion_df.Subject == subject],
            id_vars=["Block"],
            value_vars=["Reported", "Decoded"],
        )
        plt.rcParams["figure.figsize"] = [8, 6]
        fig, ax = plt.subplots()
        plt.title("Reported and decoded intrusions")
        sns.lineplot(
            data=df,
            x="Block",
            y="value",
            hue="variable",
            ci=68,
            legend="full",
            linewidth=5,
            marker="o",
            markersize=14,
        )
        plt.ylabel("Proportion of intrusions")
        plt.ylim([0, 1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig(root + "Results/Decoding/" + subject + "reported-decoded.png")
        plt.close()

    # Plot averaged subjective vs decoding intrusions
    df = pd.melt(
        intrusion_df, id_vars=["Block", "Subject"], value_vars=["Reported", "Decoded"]
    )
    plt.rcParams["figure.figsize"] = [8, 6]
    fig, ax = plt.subplots()
    plt.title("Reported and decoded intrusions")
    sns.lineplot(
        data=df,
        x="Block",
        y="value",
        hue="variable",
        ci=68,
        legend="full",
        linewidth=5,
        marker="o",
        markersize=14,
    )
    plt.ylabel("Proportion of intrusions")
    plt.ylim([0.2, 0.6])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(root + "Results/Decoding/Averaged-reported-decoded.png")
    plt.close()


# %% Run intrusion decoding
if __name__ == "__main__":

    count_intrusions(exclude_peak=40)
