# Author: Nicolas Legrand (legrand@cyceron.fr)

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mne.stats import permutation_cluster_1samp_test, permutation_t_test
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.forest import RandomForestClassifier
from scipy.ndimage.filters import gaussian_filter1d
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

import os

task = "Attention"
root = "E:/EEG_wd/Machine_learning/"
Names = os.listdir(root + task + "/1_raw")  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)

root = "E:/EEG_wd/Machine_learning/"

# =======================
# %% Decoding - Attention
# =======================
def run_decoding_attention(subject, classifier):
    """
    Run a sliding decoder on the Attention task and return the 10 fold AUC scores.

    Parameters
    ----------
    * subject: str
        subject reference (e.g. '31NLI')
    * classifier: sklearn object
        Define the ML kernel to use.

    Return
    ------
    * scores: numpy array
        ROC-AUC scores, time * 10 fold CV.

    References
    ----------
    [1]: https://mne-tools.github.io/stable/auto_examples/decoding/plot_decoding_spatio_temporal_source.html
    """

    # Import behavioral and EEG data.
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    # Define features and labels
    data = attention._data[attention_df.Cond1 != "Think", :, :]
    labels = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

    # Create classification pipeline based on the provided classifier.
    clf = make_pipeline(StandardScaler(), classifier)

    # Create the sliding classifier
    time_gen = SlidingEstimator(clf, scoring="roc_auc", n_jobs=1)

    # Run a 10 fold cros validated classification
    scores = cross_val_multiscore(time_gen, data, labels, cv=8)

    # Plot single subject decoder performances
    sns.set_context("talk")
    df = pd.DataFrame(scores).melt()
    df["Time"] = (df.variable / 100) - 0.2

    # Plot scores
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Time", y="value", data=df, ci=68, color="royalblue")
    plt.axhline(y=0.5, linestyle="--", color="gray")
    plt.axvline(
        x=(scores[:, 40:70].mean(0).argmax() / 100) + 0.2, color="red", linestyle="--"
    )
    plt.axvline(x=0, color="k")
    plt.ylabel("AUC")
    plt.ylim([0.4, 1.0])
    plt.axvspan(0.2, 0.5, alpha=0.1, color="red")
    plt.annotate("Window of interest", xy=(0.1, 0.90))
    plt.savefig(root + "Results/Decoding/" + subject + "decoding.png")
    plt.clf()
    plt.close()

    return np.asarray(scores)


# =============================================================================
# %% Sliding classifier - Attention
# =============================================================================

if __name__ == "__main__":

    scores = []
    for subject in Names:

        scores.append(run_decoding_attention(subject, classifier).mean(0))

    np.save(root + "Attention_decoding.npy", np.asarray(scores))

    scores = np.load(root + "Attention_decoding.npy")

    df = pd.DataFrame(scores).melt()
    df["Time"] = (df.variable / 100) - 0.2

    # Permutation 1 sample
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        scores - 0.5, threshold=None, n_permutations=5000, tail=0, n_jobs=1
    )

    # Create new stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.01:
            T_obs_plot[c] = T_obs[c]

    # Plot scores
    plt.rcParams["figure.figsize"] = [12, 6]
    sns.set_context("talk", font_scale=1.4)
    sns.lineplot(x="Time", y="value", data=df, ci=68, color="royalblue")
    plt.axhline(y=0.5, linestyle="--", color="gray")
    plt.axvline(x=0.2, color="red", linestyle="--")
    plt.axvline(x=0.0, color="k")
    plt.ylabel("AUC", size=30)
    plt.xlabel("Time (s)", size=30)
    plt.xticks(np.arange(-0.2, 1.5, 0.2))

    # plot significant time range
    plt.fill_between(
        np.arange(-0.2, 1.51, 0.01),
        0.5,
        scores.mean(0),
        where=~np.isnan(T_obs_plot),
        color="b",
        alpha=0.2,
    )

    mask = np.ma.masked_where(np.isnan(T_obs_plot), scores.mean(0))

    plt.plot(np.arange(-0.2, 1.51, 0.01), mask, color="k", linewidth=3)
    sns.despine(offset=10, trim=True)
    plt.savefig("attention_decoding.svg", dpi=300)
    plt.clf()
    plt.close()


# =============================================================================
# %% Topomaps
# =============================================================================

from scipy.stats import trim_mean

trim = lambda x: trim_mean(x, 0.1, axis=0)

total = []
for subject in Names:

    # Import EEG data.
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")

    baseline = (
        attention[attention_df.Cond1 == "Baseline"]
        .average(method=trim)
        .apply_baseline(baseline=(-0.2, 0))
    )
    intrusion = (
        attention[attention_df.Cond1 == "No-Think"]
        .average(method=trim)
        .apply_baseline(baseline=(-0.2, 0))
    )

    total.append(intrusion._data - baseline._data)

np.save(root + "Attention_topomap.npy", np.asarray(total))

data = np.load(root + "Attention_topomap.npy")

fig, axs = plt.subplots(1, 7, figsize=(15, 5), facecolor="w", edgecolor="k")
fig.subplots_adjust(hspace=0.5, wspace=0.001)

axs = axs.ravel()

for i, rg in enumerate(np.arange(48, 54, 1)):

    # Load data
    this_data = data[:, :, rg]

    n_permutations = 50000
    T0, p_values, H0 = permutation_t_test(this_data, n_permutations, n_jobs=1)

    # Extract mask and indices of active sensors in the layout
    mask = p_values[:, np.newaxis] <= 0.05

    evoked = mne.EvokedArray(T0[:, np.newaxis], attention.average().info, tmin=0.0)

    evoked.plot_topomap(
        ch_type="eeg",
        times=0,
        scalings=1,
        time_format=None,
        cmap=plt.cm.get_cmap("RdBu_r", 12),
        vmin=-6.0,
        vmax=6,
        units="t values",
        mask=mask,
        axes=axs[i],
        size=3,
        show_names=lambda x: x[4:] + " " * 20,
        time_unit="s",
        show=False,
    )

plt.savefig("attention_topomaps.svg", dpi=300)
plt.clf()
plt.close()
