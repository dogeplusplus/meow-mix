import os
import logging
import subprocess
import multiprocessing

from os.path import basename
from redvid import Downloader
from psaw import PushshiftAPI

DATA_DIR = "data/reddit"
VIDEO_DIR= f"{DATA_DIR}/videos"
AUDIO_DIR = f"{DATA_DIR}/audio"

def scrape_audio(subreddit, target_dir, limit=100):
    """Scrape videos from a specific subreddit.

    Args:
        subreddit (str): subreddit to search
        target_dir (str): folder to store outputs
        limit (int): maximum number of videos to try
    """
    os.makedirs(target_dir, exist_ok=True)
    api = PushshiftAPI()
    video_prefix = "https://v.redd.it"

    results = api.search_submissions(
        subreddit=subreddit,
        limit=limit,
        filter=["url"],
    )

    for post in results:
        if post.url.startswith(video_prefix):
            downloader = Downloader(post.url, DATA_DIR, min_q=True)
            try:
                downloader.download()
            except BaseException:
                    logging.error(f"Could not download: {post.url}. Continuing")


def extract_audio(directory, target_dir):
    """Change mp4 video files to audio.

    Args:
        directory (str): path to folder containing mp4 files.
        target_dir (str): path to store converted audio files.
    """
    os.makedirs(target_dir, exist_ok=True)
    threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=threads)
    work_list = [
        [
            "ffmpeg",
             "-i",
             f"{directory}/{basename(file)}",
             "-ac",
             "2",
             "-f",
             "wav",
             f"{target_dir}/{basename(file).replace('.mp4', '.wav')}"
        ]
        for file in os.listdir(directory)
    ]
    pool.map(subprocess.run, work_list)


def main():
    subreddit = "catswhoyell"
    scrape_audio(subreddit)
    extract_audio(VIDEO_DIR)


if __name__ == "__main__":
    main()
