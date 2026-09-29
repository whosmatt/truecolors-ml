# truecolors-ml

Turns out audio reactive effects are really tricky to do well and also a bit of an unsolved problem.  
This repo contains the training pipeline for the main [truecolors](https://github.com/whosmatt/truecolors) project.  
Target is an ESP32-S3 with a measured budget of ~150k int8 MACs per 512 sample (48kHz) audio block.   
The python implementation in this repo is AI slop, don't expect high code quality.  

The goal is reconstructing the metronome (more accurately: the beat without meter) to a live track, with phase within ~30ms.  
It is much more satisfying than a traditional (multiband) RMS based effect, but also much more fragile. A slight mismatch is immediately noticeable.

## Beating BeatNet

[BeatNet](https://github.com/mjhydri/BeatNet) is a modern net with the same capability of predicting beats in an audio stream.  
When ran on my test set, my model outperforms BeatNet on tempo&phase at 1/5th the size and 1/3th the compute, most of which I attribute to clean training data. 

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

#### Vague model history

- Pure drum classifier & discrete tracking
  - Trained on drum samples and tested on drumloops
    - Mediocre performance and poor generalization, even on drumloops
- Beat prediction (beat + beat_offset)
  - Trained on synthetic drumloops & discrete tracking
    - Extremely high phase precision on real drumloops
    - Poor generalization on real music
  - Trained on synthetic music, built from midi, drum samples and melody loops
    - Mediocre generalization, still good phase precision
- Beat prediction on real music & discrete tracking
  - Trained on manually labeled real music, bootstrapped by the earlier high phase precision model
  - Good generalization, good phase precision (although reduced)
  - Tracking not great, easily thrown off by a breakdown
  - Significant improvement by using mel, which was previously worse when only relying on drums
- Two stage beat prediction on real music & discrete tracking
  - Above model but with another stage added for longer context
  - Simply raising coarse context did not do much, but adding a second stage to the model to clean up beat activations brought a substantial improvement
  - Second stage uses up additional compute, but still well within budget
  - Trained on playlists with sudden track changes for quick recovery
  - Solid performance, beats BeatNet

## Current best approach: Beat prediction and ML tracking

### Dataset

Training corpus comes from my full sample library, ground truth from Ableton Live's built-in classifier.  
After careful filtering, this leaves 7560 kicks, 11274 snares/claps/rims/snaps, 4651 hats, 6806 percussion, 6971 drum loops, and 1123 midi drum loops which can be combined with hits for synthetic fully tagged data.  
Additionally, 3731 melodic loops and 2144 non-melodic textures.  
I sampled 100 examples from each category to judge quality and found zero misclassifications.  

For better generalization, a yt-dlp ingestion pipeline is present. I used it to add 293 handpicked tracks with challenging rhythms and a large genre variety. These tracks were manually labeled using the [slop labeling tool](./labeler/), as close to millisecond precision as possible.  
Due to the relatively small size of this set, multiple tracks from the same artist are kept grouped because of their similarity.

One notable finding: Something on the way from DAW to youtube often stretches the audio slightly, so assuming that BPM should be a whole number is not reliable. Many songs will have be somewhere around ~0.05 BPM off, which would throw alignment off when rounded.  

I can't share the dataset due to license restrictions.  

### Preprocessing

I recorded an IR of the target mic against a calibrated speaker inside a typical room, which is then applied to the training data.
This will do for now, but a future plan is to split the mic response from the room response and apply a variety of room IRs for augmentation.  
Coil whine is included via recordings that are mixed into the training samples. They contain the mic noise floor too.  
Silent attack is stripped from samples to reduce onset variance.  
The entire on-device preprocessing chain is compiled, wrapped into a python module and applied to the training data verbatim, this includes the lowpass filter. 

### Summary

#### Model architecture
Two dense ReLU MLPs in series, quantized to int8:
- Stage A (528, 128, 64) with a single beat head
- Stage B (528, 64, 32), same input plus beat output from stage A, with four int8 sigmoid heads:
  - beat: presence of a beat in the current block
  - beat_offset: exact beat position within the block
  - hit x4 (kick, snare, hihat, none): presence of the corresponding hit in the current block, used as training aid and for diagnostics
  - music: presence of music for noise rejection

Each stage starts with a learned linear projection (28 features to 12) that runs once per block. This allows both stages to have 528 inputs despite stage B having extra features.

113k MACs per block, about 3/4 of the measured truecolors compute budget.  
Quantized to int8 tflite: 84kB + 43kB

Designed for a discrete stage 2 with autocorrelation and phase comb fit. 

#### Input
- 12 FE features + 16 mel flux bands per block at 93.75 blocks/s
- 2.87s of context as a three-resolution input with raw and averaged projected samples
  - dense: 16 blocks including 2 blocks lookahead
  - medium: 16 means of 4 blocks
  - coarse: 12 means of 16 blocks
- Stage B receives stage A beat output, delayed by 3 blocks as quasi-lookahead

#### Results
Results: [9-mel-and-songs](./results/9-mel-and-songs/README.md), [10-two-stage](./results/10-two-stage/README.md)  
Builds on strategy 8, adds mel flux to the input and a second stage.

In addition to synthetic loops, beat-labeled drum and tempo-labelled melodic loops, this approach uses 293 manually labeled tracks, both as whole songs and snippets with sudden cuts/crossfades.  
Mel brought a clear improvement at no extra compute, mostly on melodic material.  
Using a second stage to clean up the beat predictions improved tempo accuracy a lot, with less triplet and octave failures. The attempt to achieve the same via feedback in a single model didn't work.  
Due to the small music set, performance on real music is measured with 5x cross-validation, grouped by artist:  
72.7% of 12s windows are usable, compared to the previous 64.9%.  
Median recovery time on tempo/track changes is 5.3s.  
While the frontend comb filter was dropped (partially because the coil whine was greatly reduced in hardware), removing or changing the lowpass degraded beat prediction performance in all testing so far.  
Beat detection performance is rock solid on 4-on-the-floor genres and performs reliably on other common rhythmic patters. It does work on drum breaks to some extent but not as reliably.  
Music detection is not performing well yet.

Wider models and longer context were retested with real music but did not bring a clear improvement. Causal TCN, GRU, as well as sparse context max-pooling and peak-picking were tested but did not outperform the current approach. See [6-context-and-sequence](./results/6-context-and-sequence/README.md).
