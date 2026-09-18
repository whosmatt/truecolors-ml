# truecolors-ml

Turns out audio reactive effects are really tricky to do well and also a bit of an unsolved problem.  

## Approaches

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

### ML kick/snare classifier and autocorrelation

WIP  
Same as above but with an ML classifier.
A much bigger dataset is planned by using Ableton Live's classifier as a teacher model.

Start with [HANDOFF.md](HANDOFF.md).

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
