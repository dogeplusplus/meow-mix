import torch
import librosa
import numpy as np
import mlflow.pytorch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from librosa.feature import mfcc



def predict(model, x):
    n_mfcc = 40
    sample_rate = 22050

    mel_coefficients = mfcc(x, sample_rate, n_mfcc=n_mfcc)
    time_frames = mel_coefficients.shape[-1]
    chunks = int(np.ceil(time_frames / 160))

    pad_size = (chunks * 160) - time_frames
    mfcc_pad = np.pad(mel_coefficients, ((0, 0), (0, pad_size)), mode="wrap")
    mfcc_split = np.split(mfcc_pad, 160, axis=1)

    mfcc_batch = np.stack(mfcc_split)
    probabilities = model(torch.FloatTensor(mfcc_batch).to("cuda"))
    classifications = torch.argmax(probabilities, dim=-1)
    sequence = torch.reshape(classifications, (1, -1))
    original_length = x.size

    predictions = F.interpolate(sequence.unsqueeze(0).to(torch.float), original_length, mode='linear')
    predictions = torch.squeeze(predictions).to(torch.int64)

    return predictions


def main():
    model_path = "mlruns/0/ec90ba732dec470492d3d6e7089e644b/artifacts/model"
    model = mlflow.pytorch.load_model(model_path)

    audio, _ = librosa.load("data/reddit/1vtdusf08us21-DASH_240.wav")
    predictions = predict(model, audio)
    predictions = predictions.cpu().detach().numpy()
    fig, ax = plt.subplots()

    ax.plot(audio)
    ax.fill_between(np.arange(len(predictions)), 0, 1, color="green", where=predictions==1, alpha=0.3, transform=ax.get_xaxis_transform())
    ax.fill_between(np.arange(len(predictions)), 0, 1, color="red", where=predictions==2, alpha=0.3, transform=ax.get_xaxis_transform())

    plt.show()

if __name__ == "__main__":
    main()
