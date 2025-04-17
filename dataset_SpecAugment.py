from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms as T

import os
from pathlib import Path
from typing import Tuple, Union, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
from torchaudio.transforms import Resample, Spectrogram, TimeStretch, FrequencyMasking, TimeMasking, MelScale

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
SAMPLE_RATE = 16000
_DATA_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}

class LibriSpeechAugmented(LIBRISPEECH):
    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        augmentations: Callable = None,
        max_length: int = 10 * 16000,
        input_freq=16000,
        resample_freq=8000,
        n_fft=1024,
        n_mel=256,
    ) -> None:
        super(LibriSpeechAugmented, self).__init__(
            root,
            url,
            folder_in_archive,
            download)
        self.augment = augmentations
        self.max_length = max_length
        if not self.augment:
            self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)
            self.spec = Spectrogram(n_fft=n_fft, power=2)
            self.mel_scale = MelScale(n_mels=n_mel)
            self.max_spec_l = 1 + (max_length - n_fft) // (n_fft // 2)
        else:
            self.max_spec_l = 1 + (max_length - augmentations.n_fft) // (augmentations.n_fft // 2)

    def __getitem__(self, index: int):
        (waveform,
         sample_rate,
         transcript,
         speaker_id,
         chapter_id,
         utterance_id) = super().__getitem__(index)
        #print("waveform shape:", waveform.shape)

        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            padding = torch.zeros((waveform.shape[0], self.max_length - waveform.shape[1]))
            #print("padding shape:", padding.shape)
            waveform = torch.cat((waveform, padding), dim=1)
        #print("after padding shape:", waveform.shape)

        if self.augment:
            spec = self.augment(waveform)
        else:
            resample = self.resample(waveform)
            spec = self.spec(resample)[0]
            spec = spec.abs().pow(2)
            spec = self.mel_scale(spec)

        if spec.shape[1] > self.max_spec_l:
            spec = spec[:, :self.max_spec_l]
            length = spec.shape[1]
        else:
            length = spec.shape[1]
            padding = torch.zeros((spec.shape[0], self.max_spec_l - spec.shape[1]))
            #print("padding 2 shape:", padding.shape)
            spec = torch.cat((spec, padding), dim=1)

        return spec.T, length

class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=8000,
        n_fft=1024,
        n_mel=256,
    ):
        super().__init__()

        self.n_fft = n_fft

        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = Spectrogram(n_fft=n_fft, power=None)

        self.time_stretch = TimeStretch(
            n_freq=n_fft//2+1,
            hop_length=n_fft//2,
            fixed_rate=None)
        self.spec_aug = torch.nn.Sequential(
            FrequencyMasking(freq_mask_param=16),
            TimeMasking(time_mask_param=16),
        )

        self.mel_scale = MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(resampled)[0]
        spec = self.time_stretch(spec, torch.rand(1).item()*0.4+0.8)
        spec = spec.abs().pow(2)
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel

class LibriSpeechWav2Vec(LIBRISPEECH):
    def __init__(
            self,
            root: Union[str, Path],
            url: str = URL,
            folder_in_archive: str = FOLDER_IN_ARCHIVE,
            download: bool = False,
            max_length: int = 10 * 16000,
            input_freq=16000,
            resample_freq=8000,
            n_fft=1024,
            n_mel=256,
    ) -> None:
        super(LibriSpeechWav2Vec, self).__init__(
            root,
            url,
            folder_in_archive,
            download)

        self.max_length = max_length
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        self.max_spec_l = 1 + (max_length - n_fft) // (n_fft // 2)
        self.n_fft = n_fft

    def __getitem__(self, index: int):
        (waveform,
         sample_rate,
         transcript,
         speaker_id,
         chapter_id,
         utterance_id) = super().__getitem__(index)
        # print("waveform shape:", waveform.shape)

        length = min(1 + (waveform.shape[1] - self.n_fft) // (self.n_fft // 2), self.max_spec_l)

        resample = self.resample(waveform)
        # print("res shape:", resample.shape)
        spec = self.spec(resample)[0]
        spec = spec.abs().pow(2)

        spec = self.mel_scale(spec)

        if spec.shape[1] > self.max_spec_l:
            spec = spec[:, :self.max_spec_l]
        else:
            padding = torch.zeros((spec.shape[0], self.max_spec_l - spec.shape[1]))
            spec = torch.cat((spec, padding), dim=1)

        return spec.T, length