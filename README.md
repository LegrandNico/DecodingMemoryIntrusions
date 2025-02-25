
[![DOI](https://zenodo.org/badge/180201041.svg)](https://zenodo.org/badge/latestdoi/180201041)

# Decoding memory intrusions from early visual interference

Data and code for the following paper:

> Legrand, N., Etard, O., Viader, F., Clochon, P., Doidy, F., Eustache, F., & Gagnepain, P. (2022). Attentional capture mediates the emergence and suppression of intrusive memories. IScience, 25(12), 105516. https://doi.org/10.1016/j.isci.2022.105516

## Data

Some of the data accompagning the paper are stored in the `data` folder:

* `behavior.txt`

* `merged.txt`

* `metacognition.txt`

## Code

Scripts and jupyter notebooks are stored in the `code` folder.

### Notebooks

* `Figures.ipynb` contains  goupd level summary analysis and figures reported in the paper.

### Scripts

* `1_run_filter.py` .

* `2_run_epochs.py` .

* `3_run_ica.py` .

* `4_run_autoreject.py` .

* `5_decimate.py` .

* `6_Attention_raw_decoding.py` .

* `7_Attention_time_frequency_decoding.py` .

* `8_Intrusions_Shuffled_Labels.py` .

## Figures

### Figure 1

![Figure 1: ](./figures/Figure1.png)
**Figure 1. Experimental design and behavioural results. A.** [left panel] During the attentional procedure, the participant was instructed to categorize visual stimuli as fast as possible. Each trial started with the picture of an object presented at the centre of the screen, surrounded by either a green or a red box. A larger image appeared in the background 200 milliseconds after the trial started for Exploration (green trials) and Intrusion conditions (red trials), but not after No intrusion condition (red trials). For green trials, the participant had the instruction to explore this image and to indicate whether living beings were present in the picture or not. For red trials, participants were instructed to stay focused on the central image and to indicate whether the depicted image corresponded to a living or non-living thing. As for No-Think trials during the TNT task, participants were instructed to try their best to block the images from entering their consciousness and to maintain their attention on the central object. If the background scene penetrated their consciousness, participants were told to push it out of their minds and that they should fixate and concentrate on the object-cue. An image appeared in the background 200ms after the onset of the central image for only one-half of the “red” trials (Intrusive condition). The other half corresponded to the presentation of the target object without distracting the scene (Non-intrusive condition). [right panel] Response time associated with Intrusion, Non-Intrusion, and Exploration conditions. Overall, the speed of responses increases across the eight experimental blocks, showing the presence of a task-learning effect. However, within red trials for which the task consists in focusing on the central cue, Intrusion conditions were associated with longer response times compared with Non-Intrusion, showing the presence of greater image interference during this condition. The shaded area represents the bootstrapped 95% CI. **B.** [left panel] Memory suppression task (“Think/No-Think” paradigm). After the initial learning of object-scenes pairs (not displayed here), Think and No-Think items were presented with green or red boxes, respectively, during the TNT phase (see main text for a detailed description of the instruction). [right panel] Same-probe recall of the associated scenes after the TNT procedure. Memory performances for No-Think items were reduced compared to both Think and Baseline cues. The error bars represent the bootstrapped 95% CI.

### Figure 2

![Figure 2: ](./figures/Figure2.png)
**Figure 2. Decoding performances during the attention task. A.** Random forest classifiers had performances significantly higher than chance from 250 ms to 580 ms following the onset of the initial central target image (the intrusive background image appears at 200 ms and is marked with a red dashed line; see Figure 1a for details on the attention task). The topographic maps in the lower part represent the contrast between Intrusion and Non-intrusion conditions. Higher decoding scores were associated with a significantly higher electric activity over occipital electrodes, coupled with a significant reduction of activity over frontal electrodes. **B.** Decoding performances across time and frequencies during the attention task. To discriminate the contribution of different frequency bands supporting higher classification accuracy, we applied the decoding approach to all frequency bands between 3 and 30 Hz. The central panel shows the AUC scores for each time and frequency point. The black lines highlight the contours of the significant clusters revealed by the one-sample cluster permutation test (20000 permutations). We found an early significant increase in decoding performance centred around the Theta frequency range, as well as a late (700-1500 ms) increase preferentially centred around the Beta frequency range. The blue dashed line indicates the time peak of maximal decoding accuracy 160 ms after the appearance of the intrusive image on the screen. The left sub-panel shows the decoding AUC across frequencies at the highest decoding score time-point. Theta frequency (3-7 Hz) encoded more information (maximum AUC reached at 5 Hz).
