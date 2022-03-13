# Author: Nicolas Legrand (nicolas.legrand@cfin.au.dk)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
import seaborn as sns

root = "E:/EEG_wd/Machine_learning/"
names = os.listdir(root + "TNT/1_raw")  # Subjects ID
names = sorted(list(set([subject[:5] for subject in names])))

root = "E:/EEG_wd/Machine_learning/"

# =============================================================================
# %% Count EEG intrusions
# =============================================================================


def intrusions_distribution(exclude_peak):
    """
    Count and plot the number of intrusions for each time ranges.

    Parameters
    ----------
    * exclude_peak: int
        Time window to exclude before intrusions (* 10ms)

    Return
    ------
    * dist_df: pandas DataFrame
        The percentage of intrusion per time interval for each participants.
    """

    dist_df = pd.DataFrame([])
    for subject in names:

        # Load the selected classifiers
        final_df = pd.read_csv(root + "Classifiers.txt")
        time = final_df[final_df.Subject == subject].Time.iloc[0]
        length = final_df[final_df.Subject == subject].Length.iloc[0]

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba = np.load(
            root + "Results/Attention_TNT_decoding/" + subject + "_proba.npy"
        )
        tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
        tnt_df = tnt_df[tnt_df.Cond1 == "No-Think"]

        high_CI = np.load(root + "Results/Shuffled_95CI/" + subject + "-high.npy")

        data = proba[:, time, :, 1]  # Select the probabilities of an intrusions

        ci = high_CI[:, time, :]

        for ii in range(len(data)):

            if tnt_df["Black.RESP"].iloc[ii] != 1:

                # Find all the peak > 0.5
                indexes = peakutils.indexes(
                    data[ii, :], thres=0.5, min_dist=6, thres_abs=True
                )

                # Label as an intrusion if the peak > 95% CI
                dist = []  # Timing of mental events
                for id in indexes:

                    if (id > exclude_peak) & (id < 310):

                        # Check that peak > 95 CI
                        if length == 1:
                            if data[ii, id] > (ci[ii, id]):
                                dist.append(id)
                        elif length == 2:
                            if (data[ii, id] > (ci[ii, id])) & (
                                data[ii, id + 1] > (ci[ii, id + 1])
                            ):
                                dist.append(id)
                        elif length == 3:
                            if (
                                (data[ii, id] > (ci[ii, id]))
                                & (data[ii, id + 1] > (ci[ii, id + 1]))
                                & (data[ii, id + 2] > (ci[ii, id + 2]))
                            ):
                                dist.append(id)
                        elif length == 4:
                            if (
                                (data[ii, id] > (ci[ii, id]))
                                & (data[ii, id + 1] > (ci[ii, id + 1]))
                                & (data[ii, id + 2] > (ci[ii, id + 2]))
                                & (data[ii, id + 3] > (ci[ii, id + 3]))
                            ):
                                dist.append(id)
                        elif length == 5:
                            if (
                                (data[ii, id] > (ci[ii, id]))
                                & (data[ii, id + 1] > (ci[ii, id + 1]))
                                & (data[ii, id + 2] > (ci[ii, id + 2]))
                                & (data[ii, id + 3] > (ci[ii, id + 3]))
                                & (data[ii, id + 4] > (ci[ii, id + 4]))
                            ):
                                dist.append(id)
                if dist:
                    # Store only the first event timing
                    dist_df = dist_df.append(
                        pd.DataFrame(
                            {
                                "Subject": subject,
                                "Block": tnt_df["ListImage.Cycle"].iloc[ii],
                                "Emotion": tnt_df["Cond2"].iloc[ii],
                                "Time": np.min(dist),
                            },
                            index=[0],
                        ),
                        ignore_index=True,
                    )

        # Plot individual distribution
        sns.boxplot(data=dist_df[dist_df.Subject == subject], x="Block", y="Time")
        plt.savefig(root + "Results/Decoding/" + subject + "temporal-distribution.png")
        plt.close()

    dist_df.to_csv(root + "Results/Decoding/temp_dist.txt")


#        data = dist_df.groupby(['Subject', 'Block'], as_index=False).mean()
#
#
#        sns.lineplot(data=data,
#                    x='Block',
#                    y='Time')

# %% Run intrusion decoding
if __name__ == "__main__":

    intrusions_distribution(exclude_peak=40)

    intrusion_df = pd.read_csv(root + "Results/Decoding/temp_dist.txt")
    data = intrusion_df.groupby(["Subject", "Block", "Emotion"], as_index=False).mean()

    # Plot averaged temporal distribution
    plt.title("Distribution of intrusion across time")
    sns.violinplot(data=data, x="Block", hue="Emotion", y="Time")
    plt.tight_layout()
    plt.savefig(root + "Results/Decoding/Averaged-temporal-distribution.png")
    plt.close()
