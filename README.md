# EOR Tagger

EOR Tagger is a lightweight, open-source audio metadata editor and analyzer. Built for speed and simplicity as a single-file Python script, it allows you to easily edit ID3/Vorbis tags, apply metadata templates, and batch-process tracks.

Unlike standard tagging tools, EOR Tagger includes built-in algorithmic and Natural Language Processing (NLP) tools to automatically detect BPM, musical key, track energy, and lyrical sentiment.

## Features

* Batch Tagging: Select multiple tracks to apply metadata (Artist, Album, Genre, etc.) simultaneously.
* Template Stamping: Create, save, and apply custom metadata templates for regular releases.
* Cover Art Management: Embed or replace cover art (JPG/PNG) directly into your audio files.
* Audio Analytics: Automatically detect BPM, musical key, and energy/character using librosa.
* Lyrical Analysis: Extract keywords, detect language, and analyze the emotional mood of lyrics using vaderSentiment, spaCy, and rake_nltk.
* Non-Destructive Saving: Safely copies and writes metadata without altering or degrading the original audio data.
* Built-in Preview Player: Listen to your tracks and scrub through audio directly within the app.

## Installation

EOR Tagger requires Python 3.x. 

1. Clone or download this repository.
2. Install the required dependencies using pip:

Core Dependencies:
`pip install numpy soundfile mutagen Pillow sounddevice`

Analytics Dependencies (Required for BPM, Key, and Lyrical Analysis):
`pip install librosa vaderSentiment rake_nltk spacy langdetect`
`python -m spacy download en_core_web_sm`

## Usage

1. Click "+ Add Files" to import your audio files (.wav, .mp3, .flac, .aiff, .ogg).
2. Select one or multiple tracks from the library pane.
3. Edit the metadata fields, paste lyrics, or run the analytical tools.
4. Set an output folder and filename formula, then click "Save Selected" or "Save All".

## Project Status

This project was developed by Mattske (under the Koryuai tech division) as an alternative to Kid3 or other basic metadata taggers for artists & record labels.

While the code is open-source and you are welcome to use, study, or fork it for your own needs, this repository is provided as-is and is not actively seeking feature contributions or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
