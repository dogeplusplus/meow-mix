import librosa
import logging
import numpy as np
import pandas as pd

from pathlib import Path
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

@task(nout=2)
def label_audio_extraction(audio_path, label_path, n_mfcc):
    """
    Preprocess label studio data and wav files into arrays to save computational time.

    Convert waveforms into MFCCs, JSON labels into array.

    Parameters
    ----------
    audio_path: str
        Path to folder containing audio wav files
    label_path: str
        Path to the labels csv from label studio
    n_mfcc: int
        Number of frequency bands for MFCC conversion

    Returns
    -------
    mfccs: List[np.ndarray]
        List of MFCC coefficients
    labels: List[np.ndarray]
        List of labels
    """
    labels_df = pd.read_csv(label_path)
    audio_files = list(Path(audio_path).glob("**/*.wav"))

    mfccs = []
    labels = []
    for file_name in audio_files:
        waveform, sample_rate = librosa.load(file_name, sr=None)
        mel_coefficients = mfcc(waveform, sample_rate, n_mfcc=n_mfcc)

        num_frames = mel_coefficients.shape[-1]
        duration = waveform.shape[-1] / sample_rate
        raw_label = labels_df.loc[labels_df["audio"] == str(file_name), "label"].values[0]
        label = preprocess_labels(raw_label, duration, num_frames)

        mfccs.append(mel_coefficients)
        labels.append(label)

    return mfccs, labels


@task(nout=2)
def time_resolution_slicing(mfccs, labels, time_resolution):
    """
    Slice audio and label into even segments

    This function takes a fixed number of time frames and splits the label
    and audio across time into a flat list of evenly shaped audio/labels

    Parameters
    ----------
    mfccs : List[np.ndarray]
        (N_MFCC, T) List of uneven-time MFCC coefficients
    labels : List[np.ndarray]
        (T,) classification of audio (0 - background, 1 - cat, 2 - human)
    time_resolution : int
        The length of each time chunk

    Returns
    -------
    mfcc_slices: List[np.ndarray]
        Evenly cut MFCC coefficients
    label_slices: List[np.ndarray]
        Evently cut labels
    """
    mfcc_slices = []
    label_slices = []

    for mfcc, label in zip(mfccs, labels):
        time_frames = label.size
        chunks = int(np.ceil(time_frames / time_resolution))

        pad_size = (chunks * time_resolution) - time_frames
        mfcc_pad = np.pad(mfcc, ((0, 0), (0, pad_size)), mode="wrap")
        label_pad = np.pad(label, (0, pad_size), mode="wrap")

        mfcc_chunks = np.hsplit(mfcc_pad, chunks)
        label_chunks = np.split(label_pad, chunks)

        mfcc_slices.extend(mfcc_chunks)
        label_slices.extend(label_chunks)

    return mfcc_slices, label_slices


@task
def save_dataset(mfccs, labels, dest_dir):
    """
    Save dataset to directory.

    Takes the preprocessed inputs and labels and organizes them into
    a dataset directory

    Parameters
    ----------
    mfccs : List[np.ndarray]
        List of (N_MFCC, T) preprocessed MFCC coefficients
    labels : List[np.ndarray]
        List of (T,) preprocessed labels
    dest_dir : str
        Destination directory
    """
    dest_dir = Path(dest_dir)
    audio_dir = dest_dir.joinpath("audio")
    label_dir = dest_dir.joinpath("label")

    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    for i, (mfcc, label) in enumerate(zip(mfccs, labels)):
        file_name = f"{i:06d}.npy"
        audio_path = audio_dir.joinpath(file_name)
        label_path = label_dir.joinpath(file_name)

        np.save(audio_path, mfcc)
        np.save(label_path, label)


def build_flow():
    with Flow("Dataset Numpy Conversion") as flow:
        audio_path = Parameter("audio_path")
        label_path = Parameter("label_path")
        dest_dir = Parameter("dest_dir")
        n_mfcc = Parameter("n_mfcc")
        time_frames = Parameter("time_frames")

        mfccs, labels = label_audio_extraction(audio_path, label_path, n_mfcc)
        mfcc_slices, label_slices = time_resolution_slicing(mfccs, labels, time_frames)
        save_dataset(mfcc_slices, label_slices, dest_dir)

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor(scheduler="processes")
    flow.register(project_name="meow-mix")


if __name__ == "__main__":
    main()
