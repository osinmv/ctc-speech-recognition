## Description

A repo containing a set of scripts to train a speech recognition model.
You can run inference too

## Installation

``` sh
git clone git@github.com:osinmv/ctc-speech-recognition.git
cd ctc-speech-recognition
pip -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# in case you want to test on your own recording
# ffmpeg -f alsa -ac 1 -ar 16000 -i default -t 10 output.flac
python3 infer.py
```
