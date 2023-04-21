import os
import urllib.request

MODELS_DIR = "Models/"


def report(block_number, read_size, total_size):
    if block_number % 1000 == 0:
        return_to_front = '\b' * 52
        percent = round(((block_number * read_size) / total_size) * 100)
        print(f"{return_to_front}[{'█' * (percent // 2)}{'.' * (50 - (percent // 2))}]", end='')
    if block_number * read_size >= total_size:
        return_to_front = '\b' * 52
        print(f"{return_to_front}Download complete!\n")


def download_models():
    #############
    print("Downloading FastSpeech2_Meta Model")
    os.makedirs(os.path.join(MODELS_DIR, "FastSpeech2_Meta"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://huggingface.co/spaces/Flux9665/IMS-Toucan/blob/main/Models/FastSpeech2_Meta/best.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "FastSpeech2_Meta", "best.pt")),
        reporthook=report)

    #############
    print("Downloading HiFiGAN_combined")
    os.makedirs(os.path.join(MODELS_DIR, "HiFiGAN_combined"), exist_ok=True)
    filename, headers = urllib.request.urlretrieve(
        url="https://huggingface.co/spaces/Flux9665/IMS-Toucan/blob/main/Models/HiFiGAN_combined/best.pt",
        filename=os.path.abspath(os.path.join(MODELS_DIR, "HiFiGAN_combined", "best.pt")),
        reporthook=report)


if __name__ == '__main__':
    download_models()
