from pathlib import Path
from typing import List, Callable
from torch.utils.data import Dataset
import torchaudio
import torch

def clear_text(text:str)->str:
    text = text.lower()
    for punctuation in [",",".","?",":","'",'"',"!"]:
        text = text.replace(punctuation, "")
    text = text.replace(" ", "{")
    return text


class SpeechToTextDataset(Dataset):
    def __init__(self, dataset_root: Path, text_filter: Callable[[str], str] = clear_text):
        self._files: List[str] = []
        self._labels: dict[str, str] = {}
        self._dataset_root = dataset_root
        self.text_filter = text_filter
        for path, _, files in Path(dataset_root).walk():
            if not files:
                continue
            self._files.extend(
                filter(lambda name: name.endswith(".flac"), files))
            label_file = f"{path.parent.name}-{path.name}.trans.txt"
            self.__update_labels(path/label_file)
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )

    def __update_labels(self, path: Path):
        with open(path, "r") as file:
            index = 0
            for line in file.readlines():
                segment_name = f"{path.parent.parent.name}-{path.parent.name}-{index:04d}"
                text = line[len(segment_name)+1:]
                self._labels[f"{segment_name}.flac"] = self.text_filter(text) if self.text_filter else text
                index += 1

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        book_id, chapter_id, _ = self._files[idx].split("-")
        path: Path = self._dataset_root/book_id/chapter_id/self._files[idx]
        audio = torchaudio.load(path, format="flac")
        if audio[0].size()[0] == 2:
            audio = (audio[0][0].unsqueeze(0), 16000)
        text = torch.tensor([ord(c)-96 for c in self._labels[self._files[idx]]]).clamp(0,27)
        return self.transform(audio[0]).permute(0,2,1).squeeze(0), text


class TextToTextDataset(SpeechToTextDataset):
    def __init__(self, dataset_root, text_filter = clear_text):
        super().__init__(dataset_root, text_filter)
    
    def __getitem__(self, idx):
        out = torch.tensor([ord(c)-96 for c in self._labels[self._files[idx]]]).clamp(0,27)
        return out[:-1], out[1:]