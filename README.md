# truecolors-ml

Turns out audio reactive effects are really tricky to do well and also a bit of an unsolved problem.  
This repo contains the training pipeline for the main [truecolors](https://github.com/whosmatt/truecolors) project.  
Target is an ESP32-S3 with a measured budget of ~150k int8 MACs per 512 sample (48kHz) audio block.   
The python implementation in this repo is AI slop, don't expect high code quality.  

The goal is reconstructing the metronome (more accurately: the beat without meter) to a live track, with phase within ~30ms.  
It is much more satisfying than a traditional (multiband) RMS based effect, but also much more fragile. A slight mismatch is immediately noticeable.

## Previous approaches

These are approaches I've taken so far, with the last approach being current.

### RMS -> Brightness

Very simple and robust and also what pretty much every commercial product does. 
Doesn't look very satisfying, especially with dense music.  
Splitting into multiple bands can help focus on e.g. kicks but it only works well for some genres.  
Something more advanced is needed.

### Discrete kick detection

A basic algorithm looking for characteristic signs of a kick via the onset and pitch decay.  
Quite a lot better than RMS but still not very satisfying. A good audio reactive effect should be metronome synced.

### Corpus-tuned discrete kick/snare detection and autocorrelation

Similar algorithm as above but with a separate classifier for snares and hyperparameters tuned on a real dataset of kick, snares and drumloops with known BPM.  
The detected events feed into an autocorrelator that attempts to latch onto a repeating loop, thus providing BPM and an anchor.  
A PLL is used to maintain synchronization and correct the pretty awful timing inconsistency of the detection.  
The main weakness here is the poor timing accuracy and sensitivity. The autocorrelation itself is relatively robust but needs better data.

### Various ML attempts

The best run of each variant is documented in [results](./results)  
These attempts test specific approaches and are often reduced (such as music head missing). 

Some inspiration came from [Böck et al. (2012)](https://archives.ismir.net/ismir2012/paper/000049.pdf) "Evaluating the Online Capabilities of Onset Detection Methods", but most of their findings did not translate to my smaller model. 

## Current best approach: multi-head classifier and autocorrelation

### Dataset

Training corpus comes from my full sample library, ground truth from Ableton Live's built-in classifier.  
After careful filtering, this leaves 7560 kicks, 11274 snares/claps/rims/snaps, 4651 hats, 6806 percussion, 6971 drum loops, and 1123 midi drum loops which can be combined with hits for synthetic fully tagged data.  
Additionally, 3731 melodic loops and 2144 non-melodic textures.    
I can't share the dataset due to license restrictions.

### Preprocessing

I recorded an IR of the target mic against a calibrated speaker inside a typical room, which is then applied to the training data.
This will do for now, but a future plan is to split the mic response from the room response and apply a variety of room IRs for augmentation.  
Coil whine is included via recordings that are mixed into the training samples. They contain the mic noise floor too.  
Silent attack is stripped from samples to reduce onset variance.  
The entire on-device preprocessing chain is compiled, wrapped into a python module and applied to the training data verbatim, this includes the lowpass filter. 

### Summary

#### Model architecture
Dense ReLU MLP (528, 128, 64) with four int8 sigmoid heads:
- beat: presence of a beat in the current block
- beat_offset: exact beat position within the block
- hit x4 (kick, snare, hihat, none): presence of the corresponding hit in the current block, used as training aid and for diagnostics
- music: presence of music for noise rejection

76k MACs per block, about half of the measured truecolors compute budget.  
Quantized to int8 tflite: 76k weights, 86kB file

Designed for a discrete stage 2 with autocorrelation and phase comb fit. 

#### Input
- 12 FE features per block at 93.75 blocks/s
- 2.87s of context as a three-resolution input with raw and averaged raw samples
  - dense: 16 blocks including 2 blocks lookahead
  - medium: 16 means of 4 blocks
  - coarse: 12 means of 16 blocks

#### Results
Results: [8-deploy-candidate](./results/8-deploy-candidate/README.md)  
Builds on strategy 5 and 3, reuses the multi-resolution context which brought some of the biggest improvements.

This approach uses synthetic loops built from drum hits (where every hit is labelled), as well as beat-labeled drum and tempo-labelled melodic loops. 
Snares/claps/rims/snaps can be separated into individual classes but there is some overlap between them, especially snares and claps. For this run, they were combined into a single `snare` class, along `kick`, `hat` and `none`.  
Adding the `none` class brought a mild improvement in hit detection.  
While the comb filter was dropped (partially because the coil whine was greatly reduced in hardware), removing or changing the lowpass degraded beat prediction performance in all testing so far.  
Beat detection performance is rock solid on 4-on-the-floor genres and performs reliably on other common rhythmic patters. It does work on drum breaks to some extent but not as reliably.  
Music detection is not performing well yet.

Causal TCN, GRU, as well as sparse context max-pooling and peak-picking were tested but did not outperform the current approach. See [6-context-and-sequence](./results/6-context-and-sequence/README.md).
