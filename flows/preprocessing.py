import librosa
import logging
import numpy as np
import pandas as pd

from ast import literal_eval
from librosa.feature import mfcc
from prefect import Parameter, Flow, task
from prefect.executors import LocalDaskExecutor


logger = logging.getLogger(__name__)

def rename_audio_paths(labels_path):
    """
    Convert audio paths in label-studio output to use the native local path.

    Remove label-studio prefix directory and prepend the local path on disk.

    Parameters
    ----------
    labels_path: Path
        Path to the labels
    """
    df = pd.read_csv(labels_path)
    dataset_directory = labels_path.parent
    convert = lambda x: dataset_directory.join_path(x[24:])
    df["audio"] = df["audio"].apply(convert)
    df.to_csv(labels_path, index=False)


def preprocess_labels(raw_label, audio_duration, time_frames):
    """
    Parse labels into an array with class numbers.

    Quantize continuous audio labels into discrete time frames based on the MFCC spectrogram.

    Parameters
    ----------
    raw_label: str
        Raw JSON string containing labels from label-studio
    audio_duration: float
        Duration of the entire audio file in seconds
    time_frames: int
        Number of discrete frames after applying MFCC to the waveform

    Returns
    -------
    label: np.array[int]
        Array with marked labels for Background, Cat, Human (0, 1, 2)
    """
    mapping = {
        "Cat": 1,
        "Human": 2,
    }
    label = np.zeros((time_frames))
    index = lambda x: int(time_frames * x / audio_duration)

    try:
        labelled_segments = literal_eval(raw_label)
        for segment in labelled_segments:
            start = index(segment["start"])
            end = index(segment["end"])
            label[start:end] = mapping[segment["labels"][0]]
    except ValueError:
        logger.info("Skipping since could not parse label, likely to be nan.")

    return label

@task
def preprocess_dataset(audio_path, label_path, dest_dir, n_mfcc):
    """
    Preproces label studio data and wav files into arrays to save computational time.

    Convert waveforms into MFCCs, JSON labels into array.

    Parameters
    ----------
    audio_path: Path
        Path to folder containing audio wav files
    label_path: Path
        Path to the labels csv from label studio
    dest_dir: Path
        Path to store preprocessed dataset
    n_mfcc: int
        Number of frequency bands for MFCC conversion
    """
    labels = pd.read_csv(label_path)

    audio_dir = dest_dir.joinpath("audio")
    label_dir = dest_dir.joinpath("labels")

    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)


    audio_files = list(audio_path.glob("**/*.wav"))
    for file_name in audio_files:
        waveform, sample_rate = librosa.load(file_name, sr=None)
        mel_coefficients = mfcc(waveform, sample_rate, n_mfcc=n_mfcc)

        num_frames = mel_coefficients.shape[-1]
        duration = waveform.shape[-1] / sample_rate
        raw_label = labels.loc[labels["audio"] == str(file_name), "label"].values[0]
        label = preprocess_labels(raw_label, duration, num_frames)

        base_name = file_name.stem
        np.save(audio_dir.joinpath(f"{base_name}.npy"), mel_coefficients)
        np.save(label_dir.joinpath(f"{base_name}.npy"), label)


def build_flow():
    with Flow("Dataset Numpy Conversion") as flow:
        audio_path = Parameter("audio_path")
        label_path = Parameter("label_path")
        dest_dir = Parameter("dest_dir")
        n_mfcc = Parameter("n_mfcc")

        preprocess_dataset(audio_path, label_path, dest_dir, n_mfcc)

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor(scheduler="processes")
    flow.run()


if __name__ == "__main__":
    main()
