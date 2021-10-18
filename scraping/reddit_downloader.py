import os
import logging
import subprocess
import pandas as pd
import multiprocessing

from os.path import basename
from redvid import Downloader
from psaw import PushshiftAPI


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
        filter=["url", "link_flair_text"],
    )

    video_index = pd.DataFrame(columns=["video_id", "flair"])

    counter = 0
    for post in results:
        if counter >= limit:
            break
        if post.url.startswith(video_prefix):
            downloader = Downloader(post.url, target_dir, min_q=True)
            try:
                downloader.download()
                video_index.append({"video": post.url, "flair": post.link_flair_text})
                counter += 1
            except BaseException:
                    logging.error(f"Could not download: {post.url}. Continuing")

    video_index.to_csv(f"{target_dir}/index.csv")


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
            "-y",
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
    data_dir = "data/reddit"
    video_dir = f"{data_dir}/videos"
    audio_dir = f"{data_dir}/audio"

    subreddit = "catswhoyell"
    scrape_audio(subreddit, video_dir)
    extract_audio(video_dir, audio_dir)


if __name__ == "__main__":
    main()
