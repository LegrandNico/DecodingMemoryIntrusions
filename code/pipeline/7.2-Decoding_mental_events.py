# Author: Nicolas Legrand (legrand@cyceron.fr)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
import seaborn as sns

#root = "D:/EEG_wd/Machine_learning/"
root = "/home/nicolas/git/DecodingMemoryIntrusions/data/"

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


def mental_events(exclude_peak: int):
    """Count and plot the number of mental events.

    Parameters
    ----------
    exclude_peak: int
        Time window to exclude before intrusions (* 10ms).

    Returns
    -------
    The script is saving two data frames:

        * `intrusion_df`: The mean of mental events per trial averaged over experimental
        blocks (1-2, 3-4. 5-6, 7-8).
        * `intrusionTiming_df`: The timing and length of each intrusion for each
        participant. The column `Timing` log the timingof events, in miliseconds after
        the trial start.

    """

    intrusion_df = pd.DataFrame([])  # Intrusion count
    intrusionTiming_df = pd.DataFrame([])  # Intrusion timing

    # Load the classifiers
    final_df = pd.read_csv(root + "Classifiers.txt")

    for subject in names:

        for condition in ["Think", "No-Think"]:

            if len(final_df[(final_df.Subject == subject) & (final_df.Condition == condition)]) == 0:
                continue

            # Here we use the best time and lenth parameter from the intrusion detection
            # in the No-Think condition to report the performance of the same model
            # when analyzing the Think trials
            time = final_df[(final_df.Subject == subject) & (final_df.Condition == "No-Think")].Time.iloc[0]
            length = final_df[(final_df.Subject == subject) & (final_df.Condition == "No-Think")].Length.iloc[0]

            # Load probabilities for intrusions estimated by the classifier
            # trained on the Attention dataset.
            proba = np.load(
                f"{root}Results/Attention_TNT_decoding/{subject}_{condition}_proba.npy"
            )

            tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
            tnt_df = tnt_df[tnt_df.Cond1 == condition]

            high_CI = np.load(f"{root}Results/Shuffled_95CI/{condition}/{subject}-high.npy")

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
                for idx in indexes:
                    # Exclude peak < 200ms  & > 2970 ms after stim presentation
                    if (idx > exclude_peak) & (idx < 317):

                        # Check that peak > 95 CI
                        if length == 1:
                            if data[ii, idx] > (ci[ii, idx]):
                                cnt += 1
                                intrusionTiming_df = intrusionTiming_df.append(
                                    {
                                        "Subject": subject,
                                        "Condition": condition,
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "IntrusionRating": tnt_df["Black.RESP"].iloc[ii],
                                        "Timing": (idx * 10) - 200,
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
                                        "Condition": condition,
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "IntrusionRating": tnt_df["Black.RESP"].iloc[ii],
                                        "Timing": (idx * 10) - 200,
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
                                        "Condition": condition,
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "IntrusionRating": tnt_df["Black.RESP"].iloc[ii],
                                        "Timing": (idx * 10) - 200,
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
                                        "Condition": condition,
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "IntrusionRating": tnt_df["Black.RESP"].iloc[ii],
                                        "Timing": (idx * 10) - 200,
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
                                        "Condition": condition,
                                        "Emotion": tnt_df["Cond2"].iloc[ii],
                                        "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                        "Image": tnt_df["ImageFond"].iloc[ii],
                                        "IntrusionRating": tnt_df["Black.RESP"].iloc[ii],
                                        "Timing": (idx * 10) - 200,
                                        "Length": length * 10,
                                    },
                                    ignore_index=True,
                                )

                total_count.append(cnt)

            # Save predicted intrusion in df
            tnt_df["EEG_intrusions"] = np.asarray(total_count)

            # For each experimental block, count the average number of mental events
            # per trials both for Think and No-Think trials
            for block in [[1, 2], [3, 4], [5, 6], [7, 8]]:

                eeg_intrusions = tnt_df[tnt_df["ListImage.Cycle"].isin(block)][
                    "EEG_intrusions"
                ]

                count = eeg_intrusions.sum() / len(eeg_intrusions)

                intrusion_df = intrusion_df.append(
                    pd.DataFrame(
                        {
                            "Subject": subject,
                            "Condition": condition,
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
            id_vars=["Block", "Condition"],
            value_vars=["Events"],
        )

        plt.figure(figsize=(8, 6))
        plt.title("Reported and decoded intrusions")
        sns.lineplot(
            data=df,
            x="Block",
            y="value",
            hue="Condition",
            ci=68,
            legend="full",
            linewidth=5,
            marker="o",
            markersize=14,
        )
        plt.ylabel("Proportion of intrusions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{root}Results/Decoding/{subject}_mental-events.png")
        plt.close()

    intrusion_df.to_csv(f"{root}Results/Decoding/mental_events.txt")
    intrusionTiming_df.to_csv(f"{root}Results/Decoding/mental_events_timing.txt")


# %% Run intrusion decoding
if __name__ == "__main__":
    mental_events(exclude_peak=40)