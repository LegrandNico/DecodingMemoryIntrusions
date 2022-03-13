# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
import seaborn as sns

root = "D:/EEG_wd/Machine_learning/"
names = os.listdir(root + "TNT/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))


def mental_events(exclude_peak):
    """Count and plot the number of mental events.

    Parameters
    ----------
    exclude_peak: int
        Time window to exclude before intrusions (* 10ms).

    Return
    ------
    figure: Matplotlib instance

    """

    # Load the selected classifiers
    final_df = pd.read_csv(root + "Classifiers.txt")
    final_df = final_df.drop_duplicates(subset="Subject", keep="first")

    intrusion_df = pd.DataFrame([])  # Intrusion count
    intrusionTiming_df = pd.DataFrame([])  # Intrusion timing

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

            if tnt_df["Black.RESP"].iloc[ii] != 1:

                cnt = 0
                # Find all the peak > 0.5
                indexes = peakutils.indexes(
                    data[ii, :], thres=0.5, min_dist=6, thres_abs=True
                )

                # Label as an intrusion if the peak > 95% CI
                for idx in indexes:
                    # Exclude peak < 200ms after stim presentation
                    if (idx > exclude_peak) & (idx < 310):

                        # Check that peak > 95 CI
                        if length == 1:
                            if data[ii, idx] > (ci[ii, idx]):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": tnt_df["Cond1"].iloc[ii],
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "Intrusion": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )

                        elif length == 2:
                            if (data[ii, idx] > (ci[ii, idx])) & (
                                data[ii, idx + 1] > (ci[ii, idx + 1])
                            ):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": tnt_df["Cond1"].iloc[ii],
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "Intrusion": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )

                        elif length == 3:
                            if (
                                (data[ii, idx] > (ci[ii, idx]))
                                & (data[ii, idx + 1] > (ci[ii, idx + 1]))
                                & (data[ii, idx + 2] > (ci[ii, idx + 2]))
                            ):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": tnt_df["Cond1"].iloc[ii],
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "Intrusion": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )
                        elif length == 4:
                            if (
                                (data[ii, idx] > (ci[ii, idx]))
                                & (data[ii, idx + 1] > (ci[ii, idx + 1]))
                                & (data[ii, idx + 2] > (ci[ii, idx + 2]))
                                & (data[ii, idx + 3] > (ci[ii, idx + 3]))
                            ):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": tnt_df["Cond1"].iloc[ii],
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "Intrusion": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )

                        elif length == 5:
                            if (
                                (data[ii, idx] > (ci[ii, idx]))
                                & (data[ii, idx + 1] > (ci[ii, idx + 1]))
                                & (data[ii, idx + 2] > (ci[ii, idx + 2]))
                                & (data[ii, idx + 3] > (ci[ii, idx + 3]))
                                & (data[ii, idx + 4] > (ci[ii, idx + 4]))
                            ):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": tnt_df["Cond1"].iloc[ii],
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "Intrusion": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )

                total_count.append(cnt)

        # Save predicted intrusion in df
        tnt_df = tnt_df[tnt_df["Black.RESP"] != 1]
        tnt_df["EEG_intrusions"] = np.asarray(total_count)

        for block in [[1, 2], [3, 4], [5, 6], [7, 8]]:

            eeg_intrusions = tnt_df[tnt_df["ListImage.Cycle"].isin(block)][
                "EEG_intrusions"
            ]

            if len(eeg_intrusions) > 5:

                count = eeg_intrusions.sum() / len(eeg_intrusions)

                intrusion_df = intrusion_df.append(
                    pd.DataFrame(
                        {
                            "Subject": subject,
                            "Block": str(block[0]) + "-" + str(block[1]),
                            "Events": count,
                        },
                        index=[0],
                    ),
                    ignore_index=True,
                )

        # Plot averaged subjective vs decoding intrusions
        df = pd.melt(
            intrusion_df[intrusion_df.Subject == subject],
            id_vars=["Block"],
            value_vars=["Events"],
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Reported and decoded intrusions")
        sns.lineplot(
            data=df,
            x="Block",
            y="value",
            ci=68,
            legend="full",
            linewidth=5,
            marker="o",
            markersize=14,
        )
        plt.ylabel("Proportion of intrusions")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig(root + "Results/Decoding/" + subject + "mental-events.png")
        plt.close()

    intrusion_df.to_csv(root + "Results/Decoding/mental_events.txt")
    intrusionTiming_df.to_csv(root + "Results/Decoding/mental_events_timing.txt")


# %% Run intrusion decoding
if __name__ == "__main__":

    mental_events(exclude_peak=40)
