# Author: Nicolas Legrand (legrand@cyceron.fr)

import mne
import multiprocessing as mp
import numpy as np
import pandas as pd
from mne.decoding import GeneralizingEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = "/home/nicolas/git/DecodingMemoryIntrusions/data/"

# Subjects ID
names = [
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

classifier = RandomForestClassifier(
    class_weight="balanced", n_estimators=50, random_state=42
)


# =========================================
# %% Random label Decoding - Attention -> TNT
# =========================================
def shuffled_training_labels(subject: str, n_boot: int = 200) -> np.ndarray:
    """Run a generalized sliding decoder (GAT). Train on shuffled Attention labels.

    Parameters
    ----------
    subject : str
        The participant ID.
    n_boot : int
        Number of permutation.

    Returns
    -------
    ci : np.ndarray
        Upper and lower 95% CI for a noisy classifier.

    """

    # Attention data
    attention_df = pd.read_csv(root + "Attention/Behavior/" + subject + ".txt")
    attention = mne.read_epochs(root + "Attention/6_decim/" + subject + "-epo.fif")

    attention.crop(0.2, 0.5)  # Only select time window of interest to save memory

    # TNT data
    tnt_df = pd.read_csv(root + "TNT/Behavior/" + subject + ".txt")
    tnt = mne.read_epochs(root + "TNT/6_decim/" + subject + "-epo.fif")

    for condition in ["No-Think", "Think"]:

        shuffled = []

        for _ in range(n_boot):

            # Classifier
            clf = make_pipeline(StandardScaler(), classifier)

            time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1)

            X_train = attention._data[attention_df.Cond1 != "Think", :, :]
            y_train = attention_df.Cond1[attention_df.Cond1 != "Think"] == "No-Think"

            # Shuffle the trainning labels
            labels = y_train.sample(frac=1)

            # Fit the model
            time_gen.fit(X_train, labels)

            X_test = tnt._data[(tnt_df.Cond1 == condition), :, :]

            proba = time_gen.predict_proba(X_test)

            shuffled.append(proba)

        shuffled = np.asarray(shuffled)

        # 95th percentile
        ci = np.percentile(shuffled[:, :, :, :, 1], 97.5, axis=0)

        np.save(f"{root}Results/Shuffled_95CI/{condition}/{subject}-high.npy", ci)


# %% Run
if __name__ == "__main__":

    pool = mp.Pool(processes=24)
    pool.map(shuffled_training_labels, names)
    pool.close()
    pool.join()