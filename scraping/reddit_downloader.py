import os
import glob
import logging
import subprocess
import pandas as pd
import multiprocessing

from os.path import basename
from redvid import Downloader
from psaw import PushshiftAPI
from prefect import Flow, task, Parameter, mapped
from prefect.executors import LocalDaskExecutor


@task
def extract_video(subreddit, target_dir, limit):
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
        filter=["url"],
        sort_type="score",
        sort="desc"
    )

    counter = 0
    for post in results:
        if counter >= limit:
            break
        url = post.url
        if url.startswith(video_prefix):
            try:
                downloader = Downloader(url, target_dir, min_q=True)
                downloader.download()
                counter += 1
            except BaseException:
                logging.error(f"Could not download video: {url}")

    return glob.glob(f"{target_dir}/*.mp4")


@task
def convert_to_audio(video_path):
    """Change mp4 video files to audio.

    Args:
        video_path (str): list of video paths
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "2",
        "-f",
        "wav",
        video_path.replace(".mp4", ".wav")
    ]

    subprocess.run(command)
    os.remove(video_path)


def build_flow():
    with Flow("reddit-scraper") as flow:
        subreddit = Parameter("subreddit", default="catswhoyell")
        output_dir = Parameter("output_directory")
        limit = Parameter("limit", default=500)

        index = extract_video(subreddit, output_dir, limit)
        convert_to_audio(mapped(index))

    return flow


def main():
    flow = build_flow()
    flow.executor = LocalDaskExecutor(scheduler="processes")
    parameters = dict(
        output_directory="data/reddit",
        limit=500,
    )
    flow.run(parameters=parameters)


if __name__ == "__main__":
    main()
