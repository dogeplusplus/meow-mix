import matplotlib.pyplot as plt
import torchaudio
import librosa

from scipy.signal import correlate2d
from youtube_dl import YoutubeDL

video_url = "https://www.youtube.com/watch?v=DXUAyRRkI6k"
representative_meow = "data/dataset/F_SPI01_EU_MN_NAI01_102.wav"

options = {
    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
    }],
    "outtmpl": "wav.%(etx)s",
    "quiet": False
}

audio_downloader = YoutubeDL(options)
# sound = audio_downloader.download([video_url])

template_wav, template_sr = torchaudio.load(representative_meow)
video_wav, video_sr  = torchaudio.load("wav.wav")

fig, ax = plt.subplots(1, 2)
ax[0].specgram(template_wav, Fs=template_sr)
ax[1].specgram(video_wav, Fs=video_sr)
plt.show()

# plot the range of frequencies for all the meows in the clean dataset

max_frequencies = []
import glob
clean_meows = "data/dataset/*.wav"
for fp in glob.glob(clean_meows):
    waveform, sample_rate = torchaudio.load(fp)
    spec = librosa.feature.melspectrogram(waveform.numpy()[0], sample_rate)
    max_frequencies.append(spec.shape[0])

def norm(x):
    return (x - x.mean()) / x.std()

template_spec = librosa.feature.melspectrogram(template_wav.numpy()[0], template_sr)
video_spec = librosa.feature.melspectrogram(video_wav.numpy()[0], video_sr)
cross_cor = correlate2d(video_spec, template_spec, mode="valid")
plt.plot(cross_cor[0])
plt.show()
