# Save YT videos as Obsidian notes
The process looks like this:
1. Extract metadata of a YT video URL.
2. Extract the transcript.
3. Process the transcript with Gemini Flash to make more human-readable.
4. Summarize and tag the video with Gemini Pro.[^1]
5. Make a markdown note with the properties yaml at the beginning for Obsidian compatibility.

The script expects a `videos.txt` file inside `input/` where every line has a URL of a YouTube video.

Besides that, you would also need to put a `credentials.yml` file inside `config/` with the key `GEMINI_API_KEY`.

[^1]: The available tags can be modified at `config/instructions.yml`.
