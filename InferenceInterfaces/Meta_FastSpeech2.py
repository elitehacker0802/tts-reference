import os

import librosa.display as lbd
import matplotlib.pyplot as plt
import soundfile
import torch
import pickle
from functools import partial
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.ArticulatoryCombinedTextFrontend import get_language_id
from Preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor


class Meta_FastSpeech2(torch.nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        model_name = "Meta"
        language = "en"
        self.device = device
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)
        checkpoint = torch.load("Models/FastSpeech2_Meta/best.pt", map_location='cpu', pickle_module=pickle)
        self.phone2mel = FastSpeech2(weights=checkpoint["model"]).to(torch.device(device))
        self.mel2wav = HiFiGANGenerator(path_to_weights=os.path.join("Models", "HiFiGAN_combined", "best.pt")).to(torch.device(device))
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.lang_id = get_language_id(language)
        self.to(torch.device(device))

    def set_utterance_embedding(self, path_to_reference_audio):
        wave, sr = soundfile.read(path_to_reference_audio)
        self.default_utterance_embedding = ProsodicConditionExtractor(sr=sr).extract_condition_from_reference_wave(wave).to(self.device)

    def set_phonemizer_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """      
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True, silent=False)

    def set_accent_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        self.lang_id = get_language_id(lang_id).to(self.device)

    def forward(self, text, view=False, durations=None, pitch=None, energy=None):
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=True).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=self.lang_id)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
        if view:
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].set_xticks(duration_splits, minor=True)
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            ax[1].set_xticklabels(self.text2phone.get_phone_string(text))
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()
        return wave
