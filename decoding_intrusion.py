#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:38:23 2018

@author: legrand
"""
from sklearn import metrics
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import peakutils
import itertools
import numpy as np
import mne
import os
import pandas as pd

from mne.stats import permutation_t_test, permutation_cluster_1samp_test
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, accuracy_score, recall_score, average_precision_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from mne.decoding import (cross_val_multiscore, GeneralizingEstimator, LinearModel, SlidingEstimator, get_coef)
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats
from sklearn.model_selection import cross_val_score

classifier   = RandomForestClassifier(class_weight='balanced',
                                      n_estimators= 50,
                                      random_state=42)

task = 'TNT'
root = 'E:/EEG_wd/Machine_learning/'
Names = os.listdir(root + task + '/1_raw')  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

classifier   = RandomForestClassifier(class_weight='balanced',
                                      n_estimators= 50,
                                      random_state=42)

root = 'E:/EEG_wd/Machine_learning/'
    
# =========================================
# %% Decoding - Attention -> TNT
# =========================================

def run_decoding_attention_tnt(subject, classifier):
    """
    Run a generalized sliding decoder (GAT). Train on Attention, and
    predict probabilities on TNT.

    Parameters
    ----------
    * subject: str
        subject reference (e.g. '31NLI')
    * classifier: sklearn object
        Define the ML kernel to use.

    Return
    ------
    * scores: numpy array
        Probabilities for intrusion/no-intrusions * time * trials.
    * labels: True trial's labels
    References
    ----------
    [1]: https://mne-tools.github.io/stable/auto_examples/decoding/plot_decoding_time_generalization_conditions.html?highlight=generalization
    """

    # Load data from the attention task
    attention_df  = pd.read_csv(root + 'Attention/Behavior/' + subject + '.txt')
    attention     = mne.read_epochs(root + 'Attention/6_decim/' + subject + '-epo.fif')

    attention.crop(0.2, 0.5) # Only select window of interest (200ms after the intrusive image) to save memory.

    # Define training features and labels
    X_train      = attention._data[attention_df.Cond1 != 'Think', :, :]
    y_train      = attention_df.Cond1[attention_df.Cond1 != 'Think'] == 'No-Think'

    # Create classifier pipeline
    clf = make_pipeline(StandardScaler(),
                        classifier)

    # Create the GAT
    time_gen = GeneralizingEstimator(clf,
                                     scoring='roc_auc',
                                     n_jobs=1)

    time_gen.fit(X_train, y_train) # Fit the classifier to the training set.

    # Load TNT data
    tnt_df      = pd.read_csv(root + 'TNT/Behavior/' + subject + '.txt')
    tnt         = mne.read_epochs(root + 'TNT/6_decim/' + subject + '-epo.fif')

    # Define testing features and labels
    X_test      = tnt._data[(tnt_df.Cond1 == 'No-Think'), :, :]
    y_test      = tnt_df['Black.RESP'][(tnt_df.Cond1 == 'No-Think')] != 1

    out = time_gen.predict_proba(X_test) # Predict probabilities

    return out, y_test

# ===========================
# %% Attention - TNT Decoding
# ===========================

def extract_decoding(subject, overwrite = True):

    """
    Run decoding the pipeline if overwrite = True, else load the .npy file.
    """
    if overwrite:

        proba, labels = run_decoding_attention_tnt(subject, classifier)

        np.save(root + 'Results/Attention_TNT_decoding/' + subject + '_proba.npy', proba)
        np.save(root + 'Results/Attention_TNT_decoding/' + subject + '_labels.npy', labels)

    else:

        proba  = np.load(root + 'Results/Attention_TNT_decoding/' + subject + '_proba.npy')
        labels = np.load(root + 'Results/Attention_TNT_decoding/' + subject + '_labels.npy')

    return proba, labels

# =============================================================================
# %% Testing decoder
# =============================================================================

def testing_decoder(exclude_peak):
    """
    Find intrusions in the probabilities estimated by the decoders.
    Apply a gaussian filter and search peak. Grid search for threshold [0.5 - 1] and
    attention pattern [200 - 400ms after intrusive image apparition].

    Parameters
    ----------
    * exclude_peak: int
        Time window to exclude before intrusions (* 10ms)

    Return
    ------
    * final_df: Pandas DataFrame
        Best scores.

    * output_df: Pandas DataFrame
        All the classifiers.

    * cm_final: Pandas DataFrame
        Confusion matrix for the best classifiers.
    """

    output_df, cm_final = pd.DataFrame([]), []
    for subject in Names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba, labels = extract_decoding(subject, overwrite = False)

        high_CI = np.load(root  + 'Results/Shuffled_95CI/' + subject + '-high.npy')

        auc_final = 0
#        for sigma in np.arange(1, 4, 0.05): # Find the best gaussian filter to remove noise
        for time in range(5, 30): # 200 - 500 ms after intrusive image presentation

            auc_time = 0 # AUC, to be maximized
            
            if sigma is None:
                data = proba[:, time, :, 1] # Select the probabilities of an intrusions
            else:
                data = gaussian_filter1d(proba[:, time, :, 1], sigma=sigma, axis=1) # Select the probabilities of an intrusions
            ci   = high_CI[:, time, :]

            total_count = [] # Number of peaks/trials
            for ii in range(len(data)):

                cnt = 0
                # Find all the peak > 0.5
                indexes = peakutils.indexes(data[ii, :],
                                            thres=0.5,
                                            min_dist=3,
                                            thres_abs=True)

                # Label as an intrusion if the peak > 95% CI
                for id in indexes:

                    if (id > exclude_peak) & (id < 310): # Exclude peak < 400ms  & > 2900 ms after stim presentation

                        # Check that peak > 95 CI
                        if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                            cnt+=1
                total_count.append(cnt)

            pred = np.asarray(total_count) > 0 # Only prediction non-intrusion if npeaks == 0

            auc = roc_auc_score(labels, pred) # Evaluate model accuracy

            if auc > auc_time: # Only keep the best model.
                auc_time  = auc
                if auc_time > auc_final:
                    auc_final = auc_time
                    cm = confusion_matrix(labels, pred)
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            output_df = output_df.append(pd.DataFrame({'Subject'   : subject,
                                                       'Time'      : time,
                                                       'Sigma'     : sigma,
                                                       'AUC'       : auc_time}, index=[0]), ignore_index=True)
        cm_final.append(cm)
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.3,
                   vmax=0.7)
        plt.title('AUC: ' + str(auc_final), size = 20)
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Non-Intrusions', 'Intrusion'])
        plt.yticks(tick_marks, ['Non-Intrusions', 'Intrusion'], rotation = 90)
    
        fmt = '.2f'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax2 = plt.text(j, i, format(cm[i, j], fmt), size = 15, weight = 'bold',
                     horizontalalignment="center",
                     color="white" if cm[i, j] > 0.5 else "black")
    
        plt.colorbar()
        plt.ylabel('True label', size = 20, fontweight="bold")
        plt.xlabel('Predicted label', size = 20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(root + 'Results/Decoding/' + subject + 'tnt-decoding.png')
        plt.close()

    # Save results
    output_df.to_csv(root + 'raws.txt')
    np.save(root + 'Confusions.npy', np.asarray(cm_final))

    # Select best decoders
    idx = output_df.groupby(['Subject'])['AUC'].transform(max) == output_df['AUC']
    final_df = output_df[idx]

    final_df.to_csv(root + 'Classifiers.txt')

    return final_df, output_df, cm_final

# %% Run intrusion decoding
if __name__ == '__main__':
    
    testing_decoder(exclude_peak=40)
    
    # Plot averaged confusion matrix 
    cm = np.load(root + 'Confusions.npy').mean(0)
    final_df = pd.read_csv(root + 'Classifiers.txt')
    final_df = final_df.drop_duplicates(subset='Subject', keep='first')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.3,
                   vmax=0.7)
    plt.title('AUC: ' + str(final_df.AUC.mean()), size = 20)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Intrusions', 'Intrusion'])
    plt.yticks(tick_marks, ['Non-Intrusions', 'Intrusion'], rotation = 90)

    fmt = '.2f'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax2 = plt.text(j, i, format(cm[i, j], fmt), size = 15, weight = 'bold',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 0.5 else "black")

    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label', size = 20, fontweight="bold")
    plt.xlabel('Predicted label', size = 20, fontweight="bold")
    plt.savefig(root + 'Results/Decoding/Average-tnt-decoding.png')
    plt.close()

    # AUC stripplot
    sns.stripplot(data=final_df, x='AUC')

# =============================================================================
# %% Topomap of selected patterns
# =============================================================================

def topomap_best():
    """
    Plot the averaged topomap of selected attentional patterns 
    to decode intrusion.

    Parameters
    ----------


    Return
    ------
    * Matplotlib instance
    """

    # Load the selected classifiers
    final_df = pd.read_csv(root + 'Classifiers.txt')
    final_df = final_df.drop_duplicates(subset='Subject', keep='first')
    
    gini = []
    for subject in Names:
        
        time = final_df[final_df.Subject == subject].Time.iloc[0] + 40
        
        # Import behavioral and EEG data.
        attention_df = pd.read_csv(root + 'Attention/Behavior/' + subject + '.txt')
        attention = mne.read_epochs(root + 'Attention/6_decim/' + subject + '-epo.fif')
    
        # Define features and labels
        data = attention._data[attention_df.Cond1 != 'Think',:,time]
        labels = attention_df.Cond1[attention_df.Cond1 != 'Think'] == 'No-Think'
    
        # Create classification pipeline based on the provided classifier.
        scaler = StandardScaler()
        data = scaler.fit_transform(X=data)
    
        classifier.fit(data, labels)
        gini.append(stats.zscore(classifier.feature_importances_))
        

attention = attention.average()

# set cluster threshold
tail = 0 # 0.  # for two sided test
p_thresh = 0.01 / (1 + (tail == 0))
n_samples = len(total_contrast)
threshold = -stats.t.ppf(p_thresh, n_samples - 1)

# Make a triangulation between EEG channels locations to
# use as connectivity for cluster level stat
connectivity = mne.channels.find_ch_connectivity(attention.info, 'eeg')[0]

cluster_stats = permutation_cluster_1samp_test(np.asarray(gini) - (1/102),
                                               threshold=threshold,
                                               verbose=True,
                                               connectivity=connectivity,
                                               out_type='indices',
                                               n_jobs=1,
                                               check_disjoint=True,
                                               step_down_p=0.05,
                                               seed=42)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < 0.05)[0]

# Extract mask and indices of active sensors in the layout
mask = np.zeros((T_obs.shape[0], 1), dtype=bool)
if len(clusters):
    for clus in good_cluster_inds:
        mask[clusters[clus], :] = True

evoked = mne.EvokedArray(np.asarray(gini).mean(0)[:, np.newaxis],
                         attention.info, tmin=0.)

evoked.plot_topomap(ch_type='eeg', times=0, scalings=1,
                    time_format=None, cmap=plt.cm.get_cmap('RdBu_r', 12), # vmin=0.07, vmax=0.13,
                    units='t values', mask = mask,
                    size=3, show_names=lambda x: x[4:] + ' ' * 20,
                    time_unit='s', show=False)
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
    final_df = pd.read_csv(root + 'Classifiers.txt')
    final_df = final_df.drop_duplicates(subset='Subject', keep='first')
    
    intrusion_df = pd.DataFrame([])

    for subject in Names:

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba, labels = extract_decoding(subject, overwrite = False)
        tnt_df = pd.read_csv(root + 'TNT/Behavior/' + subject + '.txt')
        tnt_df = tnt_df[tnt_df.Cond1 == 'No-Think']

        time = final_df[final_df.Subject == subject].Time.iloc[0]

        high_CI = np.load(root + 'Results/Shuffled_95CI/' + subject + '-high.npy')

        # Select the probabilities of an intrusions
        data = proba[:, time, :, 1]
        ci   = high_CI[:, time, :]

        total_count = [] # Number of peaks/trials
        for ii in range(len(data)):

            cnt = 0
            # Find all the peak > 0.5
            indexes = peakutils.indexes(data[ii, :],
                                        thres=0.5,
                                        min_dist=3,
                                        thres_abs=True)

            # Label as an intrusion if the peak > 95% CI
            for id in indexes:

                if (id > exclude_peak) & (id < 310): # Exclude peak < 400ms after stim presentation

                    # Check that peak > 95 CI
                    if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                        cnt+=1
            total_count.append(cnt)

        pred = np.asarray(total_count) > 0 # Only prediction non-intrusion if npeaks == 0

        # Save predicted intrusion in df
        tnt_df['EEG_intrusions'] = np.asarray(pred)

        for block in range(1, 9):

            for emotion in ['Emotion', 'Neutral']:

                intrusions = tnt_df[tnt_df['ListImage.Cycle'] == block]['Black.RESP'] != 1
                eeg_intrusions = tnt_df[tnt_df['ListImage.Cycle'] == block]['EEG_intrusions']

                intrusion_df = intrusion_df.append(pd.DataFrame({'Subject' : subject,
                                                           'Block' : block,
                                                           'Reported': intrusions.sum()/len(intrusions),
                                                           'Decoded': eeg_intrusions.sum()/len(eeg_intrusions)}, index=[0]), ignore_index=True)
        
        
        # Plot averaged subjective vs decoding intrusions
        df = pd.melt(intrusion_df[intrusion_df.Subject==subject], 
                     id_vars=["Block"], 
                     value_vars=["Reported", "Decoded"])
        fig, ax = plt.subplots()
        plt.title('Reported and decoded intrusions')
        sns.lineplot(data = df,
                     x = 'Block',
                     y = 'value',
                     hue='variable',
                     ci=68,
                     legend='full',
                     linewidth=5,
                     marker="o",
                     markersize=14)
        plt.ylabel('% of intrusions')
        plt.ylim([0, 1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig(root + 'Results/Decoding/' + subject + 'reported-decoded.png')
        plt.close()

    # Plot averaged subjective vs decoding intrusions
    df = pd.melt(intrusion_df, id_vars=["Block", "Subject"], 
                 value_vars=["Reported", "Decoded"])
    fig, ax = plt.subplots()
    plt.title('Reported and decoded intrusions')
    sns.lineplot(data = df,
                 x = 'Block',
                 y = 'value',
                 hue='variable',
                 ci=68,
                 legend='full',
                 linewidth=5,
                 marker="o",
                 markersize=14)
    plt.ylabel('% of intrusions')
    plt.ylim([0.2, 0.6])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(root + 'Results/Decoding/Averaged-reported-decoded.png')
    plt.close()

# %% Run intrusion decoding
if __name__ == '__main__':
    
    count_intrusions(exclude_peak=40)
    
for subject in Names:
    tnt_df = pd.read_csv(root + 'TNT/Behavior/' + subject + '.txt')
    tnt_df['Black.RESP'][tnt_df['Black.RESP'] == 3] = 4
    tnt_df['Black.RESP'][tnt_df['Black.RESP'] == 1] = 3
    tnt_df['Black.RESP'][tnt_df['Black.RESP'] == 4] = 1

tnt_df.to_csv('E:/EEG_wd/Machine_learning/TNT/Behavior/33FAM.txt')
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
    for subject in Names:
        
        # Load the selected classifiers
        final_df = pd.read_csv(root + 'Classifiers.txt')
        time = final_df[final_df.Subject == subject].Time.iloc[0]

        # Load probabilities for intrusions estimated by the classifier
        # trained on the Attention dataset.
        proba, labels   = extract_decoding(subject, overwrite = False)
        tnt_df          = pd.read_csv(root + 'TNT/Behavior/' + subject + '.txt')
        tnt_df          = tnt_df[tnt_df.Cond1 == 'No-Think']

        high_CI = np.load(root  + 'Results/Shuffled_95CI/' + subject + '-high.npy')

        data = proba[:, time, :, 1] # Select the probabilities of an intrusions

        ci      = high_CI[:, time, :]

        dist = [] # Number of peaks/trials
        for ii in range(len(data)):

            if tnt_df['Black.RESP'].iloc[ii] != 1:

                # Find all the peak > 0.5
                indexes = peakutils.indexes(data[ii, :],
                                            thres=0.5,
                                            min_dist=3,
                                            thres_abs=True)

                # Label as an intrusion if the peak > 95% CI
                for id in indexes:

                    if (id > exclude_peak) & (id < 310):

                        # Check peak > 95 CI
                        if  (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                            dist.append(id)
        if dist:
            dist = np.asarray(dist)
            for t in range(20, 320, 50):

                per = np.sum((dist > t) & (dist < t+50))/len(dist)
                dist_df = dist_df.append(pd.DataFrame({'Subject'   : subject,
                                                       'Time'     : t,
                                                       'Percentage': per}, index=[0]), ignore_index=True)
            # Plot subjective vs decoding intrusions
            plt.title('Distribution of intrusion across time')
            sns.lineplot(data = dist_df[dist_df.Subject==subject],
                         x = 'Time',
                         y = 'Percentage',
                         ci=68,
                         legend='brief',
                         linewidth=5,
                         marker="o",
                         markersize=14)
            plt.tight_layout()
            plt.savefig(root + 'Results/Decoding/' + subject + 'temporal-distribution.png')
            plt.close()

    # Plot averaged temporal distribution
    plt.title('Distribution of intrusion across time')
    sns.lineplot(data = dist_df,
                 x = 'Time',
                 y = 'Percentage',
                 ci=68,
                 legend='brief',
                 linewidth=5,
                 marker="o",
                 markersize=14)
    plt.tight_layout()
    plt.savefig(root + 'Results/Decoding/Averaged-temporal-distribution.png')
    plt.close()

# %% Run intrusion decoding
if __name__ == '__main__':
    
    intrusions_distribution(exclude_peak=40)
