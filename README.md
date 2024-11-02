# Voxit

Voxit is a simple text-to-speech and speech-to-text CLI that uses the OpenAI API to generate human-like speech or transcribe human speech to text.

> If you like this project, please consider giving it a star! :star:

## Features
- Text-to-Speech
- Speech-to-Text
- Chunked audio and text processing for large files


## Requirements
- OpenAI API key (get one [here](https://platform.openai.com/settings/organization/api-keys))
- Python `3.8` or higher (download [here](https://www.python.org/downloads/)) and add to `PATH`
- ffmpeg (download [here](https://ffmpeg.org/download.html)) and add to `PATH`

## Installation
From github:
```bash
git clone https://github.com/sean1832/voxit.git
pip install .
```

## Usage
Voxit requires an OpenAI API key to function. You can get one [here](https://platform.openai.com/settings/organization/api-keys).

> [!NOTE]
> Voxit will prompt you to enter your API key if you do not provide it as an argument.
> It will then cache the key in a file called `.openai/config` in your home directory in encrypted form for future use.
> You can delete this cache with command `voxit clear`.

### Text-to-Speech
```bash
voxit tts "input.txt" -o "output.mp3"
```

| Option           | Description                              | Default | Options                                             |
| ---------------- | ---------------------------------------- | ------- | --------------------------------------------------- |
| `input_file`     | Input file path                          | None    |                                                     |
| `-o`, `--output` | Output file path                         | None    |                                                     |
| `-k`, `--key`    | OpenAI API key                           | None    |                                                     |
| `-v`, `--voice`  | Voice name                               | `onyx`  | `alloy`, `echo`, `onyx`, `fable`, `nova`, `shimmer` |
| `-s`, `--speed`  | Speaking speed (between `0.25` to `4.0`) | `1.0`   |                                                     |



### Speech-to-Text
```bash
voxit stt "input.mp3" -o "output.txt"
```

| Option           | Description      | Default | Options |
| ---------------- | ---------------- | ------- | ------- |
| `input_file`     | Input file path  | None    |         |
| `-o`, `--output` | Output file path | None    |         |
| `-k`, `--key`    | OpenAI API key   | None    |         |

