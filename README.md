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

## Current approach: ML kick/snare classifier and autocorrelation

WIP  
Similar to above but with an ML classifier.
Training corpus comes from my full sample library, ground truth from Ableton Live's built-in classifier.  
After careful filtering, this leaves 8661 kicks, 12290 snares/claps/rims/snaps, 5169 hats, 6668 drum loops of which 985 have matching midi for full tagging and synthetic loops.  
Snares/claps/rims/snaps can be separated into individual classes but there is some overlap between them, especially snares and claps.

### Preprocessing

I recorded an IR of the target mic against a calibrated reference microphone, which is then applied to the training data.  
Silent attack is stripped from samples to reduce onset variance.  
There is no augmentation yet; adding various room IRs might be useful later.  

## Setup

WSL recommended for GPU access and access to the Ableton DB.  

```bash
sudo apt install build-essential libsndfile1
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

git submodule update --init          # firmware checkout, for the audio front end
python -m frontend.selftest          # builds the front end and verifies the binding
```

The selftest needs a C compiler and numpy.

## Front end

```python
from frontend.fe import Frontend, BLOCK_HZ

fe = Frontend(notch_hz=480)          # laser PWM frequency during capture
f = fe.run(samples_int16_48k)        # (n_blocks,) structured array, 93.75 blocks/s
f["flux"], f["level"], f["spl_db"]
```

This is the firmware's own C front end, compiled and called through ctypes, so
features are identical to what the device computes.
