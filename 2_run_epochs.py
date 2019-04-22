# Author: Nicolas Legrand (legrand@cyceron.fr)

from autoreject import Ransac
import matplotlib.pyplot as plt
import mne
import os
import pandas as pd

root = 'D:/'
Names = os.listdir(root + 'EEG/')
fname = {'eeg': '_t.fil.edf', 'eprime': '_t.txt'}
behav_var = ['Cond1', 'Cond2', 'Image.OnsetTime',
             'Black.RESP', 'ImageCentre', 'ImageFond']


def run_epochs(subject):
    """Run epochs.

    Transform raw data into epochs and match with experimental conditions.
    Reject bad epochs using a (high) threshold (preliminary rejection).

    Parameters
    ----------
    *subject: string
        The participant reference

    Save the resulting *-epo.fif file in the '3_epochs' directory.

    """
    # Load filtered data
    input_path = root + '/2_rawfilter/' + subject + '-raw.fif'
    raw = mne.io.read_raw_fif(input_path)

    # Load e-prime df
    eprime_df = root + 'EEG/' + subject + fname['eprime']
    eprime = pd.read_csv(eprime_df, skiprows=1, sep='\t')
    eprime = eprime[behav_var]

    # Revome training rows after pause for TNT
    eprime = eprime.drop(eprime.index[[97, 195, 293]])
    eprime.reset_index(inplace=True)

    # Find stim presentation in raw data
    events = mne.find_events(raw, stim_channel='STI 014')

    # Compensate for delay (as measured manually with photodiod)
    events[:, 0] += int(.015 * raw.info['sfreq'])

    # Keep only Obj Pres triggers
    events = events[events[:, 2] == 7, :]

    # Match stim presentation with conditions from eprime df
    for i in range(len(events)):
        if eprime['Cond1'][i] == 'Think':
            if eprime['Cond2'][i] == 'Emotion':
                events[i, 2] = 1
            else:
                events[i, 2] = 2
        elif eprime['Cond1'][i] == 'No-Think':
            if eprime['Cond2'][i] == 'Emotion':
                events[i, 2] = 3
            else:
                events[i, 2] = 4
        else:
            events[i, 2] = 5
    # Set event id
    id = {'Think/EMO': 1, 'Think/NEU': 2, 'No-Think/EMO': 3, 'No-Think/NEU': 4}

    # Epoch raw data
    tmin, tmax = -1.5, 4
    epochs = mne.Epochs(raw, events, id, tmin, tmax, preload=True)

    # Save epochs
    epochs.save(root + '/3_epochs/' + subject + '-epo.fif')

    epochs.info['projs'] = list()  # remove proj

    ransac = Ransac(verbose='progressbar', n_jobs=10)
    epochs_clean = ransac.fit_transform(epochs)

    evoked = epochs.average().crop(-0.2, 1.0)\
                   .apply_baseline(baseline=(None, 0))

    evoked_clean = epochs_clean.average()\
        .crop(-0.2, 1.0).apply_baseline(baseline=(None, 0))

    evoked.info['bads'] = ransac.bad_chs_
    evoked_clean.info['bads'] = ransac.bad_chs_

    # Evoked differences
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.tick_params(axis='y', which='both', left='off', right='off')

    ylim = dict(grad=(-170, 200))
    evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
    axes[0].set_title('Before RANSAC')
    evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
    axes[1].set_title('After RANSAC')
    fig.tight_layout()
    plt.savefig(root + '/3_epochs/' + subject + '-evoked.png')
    plt.clf()
    plt.close()

    # Heatmap
    ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
    fig, ax = plt.subplots(1, 1, figsize=(6, 12))
    ax.imshow(ransac.bad_log, cmap='Reds',
              interpolation='nearest')
    ax.grid(False)
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Trials')
    plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
             xticklabels=ch_names)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    fig.tight_layout(rect=[None, None, None, 1.1])
    plt.savefig(root + '/3_epochs/' + subject + '-heatmap.png')
    plt.clf()
    plt.close()

    # Save epochs
    epochs_clean.save(root + '/3_epochs/'
                      + subject + 'clean-epo.fif')


# Loop across subjects
if __name__ == '__main__':

    for subject in Names:
        run_epochs(subject)
