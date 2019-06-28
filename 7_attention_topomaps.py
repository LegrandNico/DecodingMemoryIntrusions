# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:55:41 2019

@author: nicolas
"""

import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import pandas as pd

from scipy.stats import trim_mean
trim = lambda x: trim_mean(x, 0.1, axis=0)

root = 'E:/EEG_wd/Machine_learning/'
Names = os.listdir(root + 'TNT/1_raw')  # Subjects ID
Names = sorted(list(set([subject[:5] for subject in Names])))

root = 'E:/EEG_wd/Machine_learning/'

# =============================================================================
# %% Topomap of selected patterns
# =============================================================================

"""
Plot the averaged topomap of selected attentional patterns 
to decode intrusion.

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
    gini.append(classifier.feature_importances_)
    
attention = attention.average()

# GINI index
evoked = mne.EvokedArray(stats.zscore(np.asarray(gini).mean(0))[:, np.newaxis],
                         attention.info, tmin=0.)

evoked.plot_topomap(ch_type='eeg', times=0, scalings=1,
                    time_format=None, cmap=plt.cm.get_cmap('viridis', 12), vmin=-3, vmax=3,
                    units='GINI index (z-scores)',
                    size=3, show_names=lambda x: x[4:] + ' ' * 20,
                    time_unit='s', show=False)
plt.savefig('TNT_decoding_GINI.svg', dpi=300)

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
        length = final_df[final_df.Subject == subject].Length.iloc[0]

        high_CI = np.load(root + 'Results/Shuffled_95CI/' + subject + '-high.npy')

        # Select the probabilities of an intrusions
        data = proba[:, time, :, 1]
        ci = high_CI[:, time, :]

        total_count = [] # Number of peaks/trials
        for ii in range(len(data)):

            cnt = 0
            # Find all the peak > 0.5
            indexes = peakutils.indexes(data[ii, :],
                                        thres=0.5,
                                        min_dist=5,
                                        thres_abs=True)

            # Label as an intrusion if the peak > 95% CI
            for id in indexes:

                if (id > exclude_peak) & (id < 310): # Exclude peak < 400ms after stim presentation

                    # Check that peak > 95 CI
                    if length==1:
                        if (data[ii, id] > (ci[ii, id])):
                            cnt+=1
                    elif length==2:
                        if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])):
                            cnt+=1
                    elif length==3:
                        if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                            cnt+=1
                    elif length==4:
                        if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])) & (data[ii, id+3] > (ci[ii, id+3])):
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
        plt.rcParams['figure.figsize'] = [8, 6]
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
        plt.ylabel('Proportion of intrusions')
        plt.ylim([0, 1])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig(root + 'Results/Decoding/' + subject + 'reported-decoded.png')
        plt.close()

    # Plot averaged subjective vs decoding intrusions
    df = pd.melt(intrusion_df, id_vars=["Block", "Subject"], 
                 value_vars=["Reported", "Decoded"])
    plt.rcParams['figure.figsize'] = [8, 6]
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
    plt.ylabel('Proportion of intrusions')
    plt.ylim([0.2, 0.6])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(root + 'Results/Decoding/Averaged-reported-decoded.png')
    plt.close()

# %% Run intrusion decoding
if __name__ == '__main__':
    
    count_intrusions(exclude_peak=40)

# =============================================================================
# %% Count number of mental events
# =============================================================================

def mental_events(exclude_peak):
    """
    Count and plot the number of mental events.

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
        length = final_df[final_df.Subject == subject].Length.iloc[0]

        high_CI = np.load(root + 'Results/Shuffled_95CI/' + subject + '-high.npy')

        # Select the probabilities of an intrusions
        data = proba[:, time, :, 1]
        ci = high_CI[:, time, :]

        total_count = [] # Number of peaks/trials
        for ii in range(len(data)):
            
            if tnt_df['Black.RESP'].iloc[ii] != 1:

                cnt = 0
                # Find all the peak > 0.5
                indexes = peakutils.indexes(data[ii, :],
                                            thres=0.5,
                                            min_dist=5,
                                            thres_abs=True)
    
                # Label as an intrusion if the peak > 95% CI
                for id in indexes:
    
                    if (id > exclude_peak) & (id < 310): # Exclude peak < 400ms after stim presentation
    
                        # Check that peak > 95 CI
                        if length==1:
                            if (data[ii, id] > (ci[ii, id])):
                                cnt+=1
                        elif length==2:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])):
                                cnt+=1
                        elif length==3:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                                cnt+=1
                        elif length==4:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])) & (data[ii, id+3] > (ci[ii, id+3])):
                                cnt+=1
    
                total_count.append(cnt)

        # Save predicted intrusion in df
        tnt_df = tnt_df[tnt_df['Black.RESP'] != 1]
        tnt_df['EEG_intrusions'] = np.asarray(total_count)

        for block in range(1, 9):

            for emotion in ['Emotion', 'Neutral']:

                eeg_intrusions = tnt_df[(tnt_df.Cond2==emotion) & (tnt_df['ListImage.Cycle'] == block)]['EEG_intrusions']
                
                if len(eeg_intrusions) == 0:
                    count = 0
                else:
                    count = eeg_intrusions.sum()/len(eeg_intrusions)
                    
                intrusion_df = intrusion_df.append(pd.DataFrame({'Subject' : subject,
                                                                 'Block' : block,
                                                                 'Emotion' : emotion,
                                                                 'Events': count}, index=[0]), ignore_index=True)

        # Plot averaged subjective vs decoding intrusions
        df = pd.melt(intrusion_df[intrusion_df.Subject==subject], 
                     id_vars=['Block', 'Emotion'], 
                     value_vars=['Events'])
        plt.rcParams['figure.figsize'] = [8, 6]
        fig, ax = plt.subplots()
        plt.title('Reported and decoded intrusions')
        sns.lineplot(data = df,
                     x = 'Block',
                     y = 'value',
                     hue='Emotion',
                     ci=68,
                     legend='full',
                     linewidth=5,
                     marker="o",
                     markersize=14)
        plt.ylabel('Proportion of intrusions')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig(root + 'Results/Decoding/' + subject + 'mental-events.png')
        plt.close()

    # Plot averaged subjective vs decoding intrusions
    df = pd.melt(intrusion_df, id_vars=["Block", "Emotions"], 
                 value_vars=["Events"])
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, ax = plt.subplots()
    plt.title('Number of mental events')
    sns.lineplot(data = intrusion_df,
                 x = 'Block',
                 y = 'Events',
                 hue='Emotion',
                 ci=68,
                 legend='full',
                 linewidth=5,
                 marker="o",
                 markersize=14)
    plt.ylabel('Proportion of mental events')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(root + 'Results/Decoding/Averaged-mental-events.png')
    plt.close()

# %% Run intrusion decoding
if __name__ == '__main__':
    
    mental_events(exclude_peak=40)
    
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
        length = final_df[final_df.Subject == subject].Length.iloc[0]

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
                        
                        # Check that peak > 95 CI
                        if length==1:
                            if (data[ii, id] > (ci[ii, id])):
                                dist.append(id)
                        elif length==2:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])):
                                dist.append(id)
                        elif length==3:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])):
                                dist.append(id)
                        elif length==4:
                            if (data[ii, id] > (ci[ii, id])) & (data[ii, id+1] > (ci[ii, id+1])) & (data[ii, id+2] > (ci[ii, id+2])) & (data[ii, id+3] > (ci[ii, id+3])):
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