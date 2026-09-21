# truecolors-ml

Turns out audio reactive effects are really tricky to do well and also a bit of an unsolved problem.  
This repo contains the training pipeline for the main [truecolors](https://github.com/whosmatt/truecolors) project.  
Python implementation is heavily AI-assisted, so don't expect high code quality.

## Previous approaches

These are approaches I've taken so far, with the last approach being current.

### RMS -> Brightness

Very simple and robust and also what pretty much every commercial product does. 
Doesn't look very satisfying, especially with dense music.  
Splitting into multiple bands can help focus on e.g. kicks but it only works well for some genres.  
Something more advanced is needed.

Training and model research for the audio-reactive side of
[truecolors](https://github.com/) — a learned kick/snare onset detector for the
ESP32-S3.

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

## Dataset

Training corpus comes from my full sample library, ground truth from Ableton Live's built-in classifier.  
After careful filtering, this leaves 8661 kicks, 12290 snares/claps/rims/snaps, 5169 hats, 6668 drum loops of which 985 have matching midi for full tagging and synthetic loops.  
I can't share the dataset due to license restrictions.

## Current best approach: multi-head classifier and autocorrelation

Results: [3-long-context](./results/3-long-context)

This approach only relies on individual hits and synthetic loops created from hits and midi files.  
Adding the vast amount of real drum loops (which only have period/BPM labeled and no hits) was tested in [4-join-loops](./results/4-joint-loops) but made the model perform substantially worse.  
The most likely improvement is getting more (and more diverse) midi clips into the dataset.  
Snares/claps/rims/snaps can be separated into individual classes but there is some overlap between them, especially snares and claps. For this run, they were combined into a single `snare` class, along `kick`, `hat` and `none`.  
Adding the `none` class brought a mild improvement in precision across the board.  
The autocorrelator was changed, mostly to make use of `beat-offset`, output by a regressor trained on the midi file beat positions. 

### Preprocessing

I recorded an IR of the target mic against a calibrated speaker inside a typical room, which is then applied to the training data.
This will do for now, but a future plan is to split the mic response from the room response and apply a variety of room IRs for augmentation.  
Coil whine is included via recordings that are mixed into the training samples. They contain the mic noise floor too.  
Silent attack is stripped from samples to reduce onset variance.  
The entire on-device preprocessing chain is compiled, wrapped into a python module and applied to the training data verbatim, this includes the comb and lowpass filter. 
