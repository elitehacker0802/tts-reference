import os

import gradio as gr
import numpy as np
import torch
from Utility.utils import emotion
from InferenceInterfaces.Meta_FastSpeech2 import Meta_FastSpeech2


def float2pcm(sig, dtype="int16"):
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


class TTS_Interface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Meta_FastSpeech2(device=self.device)
        self.current_speaker = "M_Angry"
        self.current_language = "English"
        self.current_accent = "English"
        self.language_id_lookup = {
            "English": "en",
            "German": "de",
            "Greek": "el",
            "Spanish": "es",
            "Finnish": "fi",
            "Russian": "ru",
            "Hungarian": "hu",
            "Dutch": "nl",
            "French": "fr",
            "Polish": "pl",
            "Portuguese": "pt",
            "Italian": "it",
        }
        self.speaker_path_lookup = {
            "M_Angry": "voices/nate_angry.mp3",
            "M_Sad": "output1.wav",
            "M_Cheerful": "voices/nate_cheerful.mp3",
            "M_Excited": "voices/nate_excited.mp3",
            "M_Friendly": "voices/robert-friendly.mp3",
            "M_Hopeful": "voices/robert-hopeful.mp3",
            "M_Normal": "voices/robert-normal.mp3",
            "M_Shouting": "output1.wav",
            "M_Terrified": "voices/nate_terrified.mp3",
            "M_Unfriendly": "voices/robert-unfriendly.mp3",
            "M_Whispering": "voices/robert-whispering.mp3",
            "F_Angry": "voices/cleo-angry.mp3",
            "F_Sad": "voices/cleo-sad.mp3",
            "F_Cheerful": "voices/cleo-cheerful.mp3",
            "F_Excited": "voices/cleo-excited.mp3",
            "F_Friendly": "voices/cleo-friendly.mp3",
            "F_Hopeful": "voices/cleo-hopeful.mp3",
            "F_Normal": "voices/cleo-normal.mp3",
            "F_Shouting": "voices/cleo-shouting.mp3",
            "F_Terrified": "voices/cleo-terrified.mp3",
            "F_Unfriendly": "voices/cleo-unfriendly.mp3",
            "F_Whispering": "voices/cleo-whispering.mp3",
        }
        self.model.set_utterance_embedding(
            self.speaker_path_lookup[self.current_speaker]
        )

    def read(self, prompt, language, accent, speaker, gender):
        language = language.split()[0]
        accent = accent.split()[0]
        if self.current_language != language:
            self.model.set_phonemizer_language(self.language_id_lookup[language])
            self.current_language = language
        if self.current_accent != accent:
            self.model.set_accent_language(self.language_id_lookup[accent])
            self.current_accent = accent
        speaker = (gender[0]) + "_" + speaker
        if self.current_speaker != speaker:
            self.model.set_utterance_embedding(self.speaker_path_lookup[speaker])
            self.current_speaker = speaker

        phones = self.model.text2phone.get_phone_string(prompt)
        if len(phones) > 1800:
            if language == "English":
                prompt = "Your input was too long. Please try either a shorter text or split it into several parts."
            elif language == "German":
                prompt = "Deine Eingabe war zu lang. Bitte versuche es entweder mit einem kürzeren Text oder teile ihn in mehrere Teile auf."
            elif language == "Greek":
                prompt = "Η εισήγησή σας ήταν πολύ μεγάλη. Παρακαλώ δοκιμάστε είτε ένα μικρότερο κείμενο είτε χωρίστε το σε διάφορα μέρη."
            elif language == "Spanish":
                prompt = "Su entrada es demasiado larga. Por favor, intente un texto más corto o divídalo en varias partes."
            elif language == "Finnish":
                prompt = "Vastauksesi oli liian pitkä. Kokeile joko lyhyempää tekstiä tai jaa se useampaan osaan."
            elif language == "Russian":
                prompt = "Ваш текст слишком длинный. Пожалуйста, попробуйте либо сократить текст, либо разделить его на несколько частей."
            elif language == "Hungarian":
                prompt = "Túl hosszú volt a bevitele. Kérjük, próbáljon meg rövidebb szöveget írni, vagy ossza több részre."
            elif language == "Dutch":
                prompt = "Uw input was te lang. Probeer een kortere tekst of splits het in verschillende delen."
            elif language == "French":
                prompt = "Votre saisie était trop longue. Veuillez essayer un texte plus court ou le diviser en plusieurs parties."
            elif language == "Polish":
                prompt = "Twój wpis był zbyt długi. Spróbuj skrócić tekst lub podzielić go na kilka części."
            elif language == "Portuguese":
                prompt = "O seu contributo foi demasiado longo. Por favor, tente um texto mais curto ou divida-o em várias partes."
            elif language == "Italian":
                prompt = "Il tuo input era troppo lungo. Per favore, prova un testo più corto o dividilo in più parti."
            phones = self.model.text2phone.get_phone_string(prompt)

        wav = self.model(phones)

        wav = emotion(wav, self.current_speaker)

        return 48000, float2pcm(wav.cpu().numpy())


meta_model = TTS_Interface()
article = "<p style='text-align: left'></a></p>"

iface = gr.Interface(
    fn=meta_model.read,
    inputs=[
        gr.inputs.Textbox(
            lines=2,
            placeholder="write what you want the synthesis to read here... \n(to prevent out of memory errors, too long inputs get replaced with a placeholder)",
            label="Text input",
        ),
        gr.inputs.Dropdown(
            [
                "English Text",
                "German Text",
                "Greek Text",
                "Spanish Text",
                "Finnish Text",
                "Russian Text",
                "Hungarian Text",
                "Dutch Text",
                "French Text",
                "Polish Text",
                "Portuguese Text",
                "Italian Text",
            ],
            type="value",
            default="English Text",
            label="Select the Language of the Text",
        ),
        gr.inputs.Dropdown(
            [
                "English Accent",
                "German Accent",
                "Greek Accent",
                "Spanish Accent",
                "Finnish Accent",
                "Russian Accent",
                "Hungarian Accent",
                "Dutch Accent",
                "French Accent",
                "Polish Accent",
                "Portuguese Accent",
                "Italian Accent",
            ],
            type="value",
            default="English Accent",
            label="Select the Accent of the Speaker",
        ),
        gr.inputs.Dropdown(
            [
                "Angry",
                "Sad",
                "Cheerful",
                "Excited",
                "Friendly",
                "Hopeful",
                "Normal",
                "Shouting",
                "Terrified",
                "Unfriendly",
                "Whispering",
            ],
            type="value",
            default="Angry",
            label="Select the Voice of the Speaker",
        ),
        gr.inputs.Dropdown(
            ["Male", "Female"], type="value", default="Male", label="Select the gender"
        ),
    ],
    outputs=gr.outputs.Audio(type="numpy", label=None),
    layout="vertical",
    title="",
    theme="default",
    allow_flagging="never",
    allow_screenshot=False,
    article=article,
)
iface.launch(enable_queue=True)
