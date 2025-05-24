from training.model import MyMediumLSTMModel
import torchaudio
import torch
import os

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

## TO RECORD ffmpeg -f alsa -ac 1 -ar 16000 -i default -t 10 output.flac


if __name__ == "__main__":
    model = MyMediumLSTMModel
    model.load_state_dict(torch.load("model/MediumLSTMModel.pth"))
    model.eval()
    model = model.to(DEVICE)
    clip = torchaudio.load(os.getenv("CLIP", 'output.flac'),format="flac")
    transform  = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    clip = transform(clip[0][0]).unsqueeze(0).permute((0,2,1)).to(DEVICE)
    logits = model(clip).softmax(dim=-1)
    characters = logits.squeeze(0)
    for i in characters:
        next_char = chr(96+int(torch.argmax(i,dim=-1).item()))
        if next_char == "`":
            continue
        elif next_char == "{":
            print(" ", end="")
        else:
            print(next_char, end = "")
    print()


