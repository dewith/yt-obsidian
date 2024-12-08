"""Take a list of youtube links and convert them to markdown notes for Obsidian."""

import json
import logging
import os
import re
import time
from datetime import UTC, datetime

import google.generativeai as genai
import yaml
from google.ai.generativelanguage_v1beta.types import content
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


def get_logger() -> logging.Logger:
    """Get a logger."""
    log_fmt = "%(levelname)s %(asctime)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    return logging.getLogger(__name__)


def fmt_time(seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    return time.strftime("%M:%S", time.gmtime(seconds))


def get_note_title(video_title: str) -> str:
    """Get a valid note title from a video title."""
    note_title = re.sub(r"[^\w\s']", "", video_title).capitalize()
    return note_title + ".md"


def get_video_links() -> list:
    """Get video links from a txt file."""
    with open("input/videos.txt", "r") as f:
        video_links = f.readlines()

    # Extract URLs from each line
    pat = re.compile(r"(https?://[^\s]+)")
    video_links = [pat.findall(line)[0] for line in video_links]

    # Remove duplicates while preserving order
    video_links = list(dict.fromkeys(video_links))
    return video_links


def create_llms() -> tuple:
    """Create the transcript and summary models."""
    with open("config/credentials.yml", "r") as f:
        credentials = yaml.safe_load(f)
    genai.configure(api_key=credentials["GEMINI_API_KEY"])

    with open("config/instructions.yml", "r") as f:
        instructions = yaml.safe_load(f)

    # Transcript model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    transcript_instruction = instructions["transcript_instruction"]
    transcript_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=transcript_instruction,
    )

    # Summary model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            enum=[],
            required=["summary", "key_takeaways", "tags"],
            properties={
                "summary": content.Schema(
                    type=content.Type.STRING,
                ),
                "key_takeaways": content.Schema(
                    type=content.Type.STRING,
                ),
                "tags": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.STRING,
                    ),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }
    summary_instruction = instructions["summary_instruction"].format(
        available_tags_str="\n".join(instructions["tags"])
    )
    summary_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        system_instruction=summary_instruction,
    )

    return transcript_model, summary_model


def get_video_data(video_link: str) -> dict:
    """Get video data from a youtube link.

    Parameters
    ----------
    video_link : str
        The youtube link to the video.

    Returns
    -------
    dict
        A dictionary containing the video data.
    """
    opts = {"quiet": True, "noprogress": True}
    with YoutubeDL(opts) as yt:
        info = yt.extract_info(video_link, download=False)
        views = info.get("view_count", 1)
        likes = info.get("like_count", 0)
        like_rate = round(100.0 * likes / views, 2)
        published = datetime.strptime(info.get("upload_date", ""), "%Y%m%d")
        video_data = {
            "id": info.get("id"),
            "title": info.get("title", ""),
            "author": info.get("channel", ""),
            "link": video_link,
            "likes": likes,
            "views": views,
            "like_rate": like_rate,
            "description": info.get("description", ""),
            "duration": round(info.get("duration", 0) / 60, 1),
            "published": published.strftime("%Y-%m-%d"),
            "created": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            "thumbnail": info.get("thumbnail", ""),
        }
    return video_data


def enrich_video_data(video_data: dict, models: tuple, logger: logging.Logger) -> dict:
    """Enrich video data with a summary and tags.

    Parameters
    ----------
    video_data : dict
        A dictionary containing the video data.
    models : tuple
        A tuple containing the models to use for generating the summary and tags.

    Returns
    -------
    dict
        A dictionary containing the video data with additional fields.
    """
    transcript_model, summary_model = models
    max_minutes_for_transcript = 15
    try:
        logger.info("| Getting transcript")
        transcript = YouTubeTranscriptApi.get_transcript(video_data["id"])
        transcript_text = "\n".join(
            [f"{fmt_time(entry['start'])} {entry['text']}" for entry in transcript]
        )
        video_data["transcript"] = transcript_text
    except Exception as e:
        logger.error("Error getting transcript: %s", e)
        video_data["transcript"] = None
    else:
        if video_data["duration"] < max_minutes_for_transcript:
            logger.info("| Processing transcript with LLM")
            chat_session = transcript_model.start_chat(history=[])
            new_transcript_text = chat_session.send_message(transcript_text).text
            video_data["new_transcript"] = new_transcript_text
        else:
            video_data["new_transcript"] = transcript_text

    try:
        logger.info("| Generating summary and tags")
        video_data_text = (
            "Title: " + video_data["title"] + "\n"
            "Author: " + video_data["author"] + "\n"
            "Transcript:\n" + video_data["new_transcript"] + "\n"
        )
        chat_session = summary_model.start_chat(history=[])
        response = chat_session.send_message(video_data_text)
        video_data.update(json.loads(response.candidates[0].content.parts[0].text))
    except Exception as e:
        logger.error("Error generating summary: %s", e)
        video_data["summary"] = video_data["description"]
        video_data["key_takeaways"] = ""
        video_data["tags"] = []

    return video_data


def save_note(video_data: dict) -> str:
    """Save the video data to a markdown note.

    Parameters
    ----------
    video_data : dict
        A dictionary containing the video data.

    Returns
    -------
    str
        The path to the saved note.
    """
    note_title = get_note_title(video_data["title"])
    note_path = os.path.join("output", note_title)
    tags_str = ""
    for tag in video_data["tags"]:
        tags_str += f'  - "{tag}"\n'

    note = (
        "---\n"
        f'title: "{video_data["title"]}"\n'
        f'source: "{video_data["link"]}"\n'
        "author:\n"
        f'- "[[{video_data["author"]}]]"\n'
        f'published: {video_data["published"]}\n'
        f'created: {video_data["created"]}\n'
        f'description: "{video_data["summary"]}"\n'
        f'like_rate: {video_data["like_rate"]}\n'
        "tags:\n"
        f"{tags_str}\n"
        "---\n\n"
        f'[![Thumbnail]({video_data["thumbnail"]})]({video_data["link"]})\n\n'
        f'{video_data["summary"]}\n\n'
        "## Key takeaways\n"
        f'{video_data["key_takeaways"]}\n\n'
        "## Transcript\n"
        f'{video_data["new_transcript"]}'
    )

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(note)

    return note_path


def main() -> None:
    """Main function."""
    logger = get_logger()
    logger.info("🔮 Saving YT videos to Obsidian notes")

    logger.info("Getting video links")
    video_links = get_video_links()

    logger.info("Creating LLM models")
    models = create_llms()

    for i, video_link in enumerate(video_links, 1):
        logger.info("Processing video %s/%s: %s", i, len(video_links), video_link)
        logger.info("| Getting video data")
        try:
            video_data = get_video_data(video_link)
        except Exception as e:
            logger.error("Error getting video data: %s", e)
            continue

        # Check if already exists
        note_title = get_note_title(video_data["title"])
        if os.path.exists("output/" + note_title):
            logger.info("| Note already exists, skipping")
            continue

        logger.info("| Enriching video data")
        video_data = enrich_video_data(video_data, models, logger)

        logger.info("| Saving note")
        note_path = save_note(video_data)
        logger.info("| Saved note to %s", note_path)


if __name__ == "__main__":
    main()
