import argparse
import getpass
import os
import shutil
from pathlib import Path

from openai import OpenAI


class ConfigManager:
    @staticmethod
    def get_key_dir() -> Path:
        if os.name == "nt":  # Windows
            key_path = Path(os.getenv("APPDATA", ""), "voxit")
        elif os.name == "posix":  # Linux
            key_path = Path("~/.local/share/voxit").expanduser()
        elif os.name == "mac":  # Mac
            key_path = Path("~/Library/Application Support/voxit").expanduser()
        return key_path

    @staticmethod
    def get_config_path() -> Path:
        return Path("~/.openai/config").expanduser()

    @staticmethod
    def save_config(config: dict, encryption: bool = False):
        import base64
        import json

        config_path = ConfigManager.get_config_path()
        data = json.dumps(config)
        # encrypt the data
        if encryption:
            key_path = ConfigManager.get_key_dir()
            key_path.mkdir(parents=True, exist_ok=True)

            # generate a new encryption key when saving or updating the API key
            encryption_key = CryptoManager.generate_key()
            data = base64.b64encode(CryptoManager.encrypt_data(data, encryption_key)).decode(
                "utf-8"
            )
            # save the encryption key
            with open(f"{key_path}/keys", "w") as f:
                f.write(encryption_key)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(data)

    @staticmethod
    def load_config(encryption: bool = False) -> dict:
        import base64
        import json

        config_path = ConfigManager.get_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                "Config file not found. Please provide an API key with -k or --key"
            )

        with open(config_path, "r") as f:
            data = f.read()
        if encryption:
            key_path = ConfigManager.get_key_dir() / "keys"
            if not key_path.exists():
                raise FileNotFoundError("Encryption key not found.")
            with open(key_path, "r") as f:
                encryption_key = f.read()
            data = CryptoManager.decrypt_data(base64.b64decode(data), encryption_key)
        return json.loads(data)


class CryptoManager:
    @staticmethod
    def encrypt_data(data: str, key: str) -> bytes:
        import base64

        import cryptography
        import cryptography.fernet

        key_byte = base64.b64decode(key)
        cipher = cryptography.fernet.Fernet(key_byte)
        return cipher.encrypt(data.encode("utf-8"))

    @staticmethod
    def decrypt_data(data: bytes, key: str) -> str:
        import base64

        import cryptography
        import cryptography.fernet

        key_byte = base64.b64decode(key)
        cipher = cryptography.fernet.Fernet(key_byte)
        return cipher.decrypt(data).decode("utf-8")

    @staticmethod
    def generate_key() -> str:
        import base64

        import cryptography
        import cryptography.fernet

        return base64.b64encode(cryptography.fernet.Fernet.generate_key()).decode("utf-8")


class OpenAIHandler:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    from typing import Literal

    def text_to_speech(
        self,
        text,
        output_path,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "onyx",
        speed: float = 1.0,
    ):
        import shutil
        import tempfile

        from pydub import AudioSegment
        from tqdm import tqdm

        if speed < 0.25 or speed > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        # Create temp directory for chunk files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            # Ensure temp directory exists
            temp_dir.mkdir(parents=True, exist_ok=True)

            chunks = Utility.split_txt_chunks(text)
            temp_files = []

            # Generate audio for each chunk with progress bar
            for i, chunk in enumerate(tqdm(chunks, desc="Generating audio chunks")):
                temp_file = temp_dir / f"chunk_{i}.mp3"
                response = self.client.audio.speech.create(
                    model="tts-1", voice=voice, input=chunk, speed=speed
                )
                # Write response directly to file
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                temp_files.append(temp_file)

            # Combine audio files
            if len(temp_files) == 1:
                # If only one chunk, just copy it to output
                shutil.copy2(str(temp_files[0]), str(output_path))
            else:
                # Combine multiple chunks
                combined = AudioSegment.from_mp3(str(temp_files[0]))
                for temp_file in temp_files[1:]:
                    audio_segment = AudioSegment.from_mp3(str(temp_file))
                    combined += audio_segment
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                combined.export(str(output_path), format="mp3")

    def speech_to_text(
        self, audio_file_path: Path, max_audio_size_mb=20, output_format="text"
    ) -> str:
        from halo import Halo
        from tqdm import tqdm

        transcriptions = []
        if audio_file_path.stat().st_size > max_audio_size_mb * 1024 * 1024:
            # Split audio into chunks
            chunks = Utility.split_audio_chunks(audio_file_path, max_audio_size_mb)
            for i, chunk in enumerate(tqdm(chunks, desc="Transcribing audio chunks")):
                response = self.client.audio.transcriptions.create(
                    model="whisper-1", response_format=output_format, file=chunk
                )
                transcriptions.append(response)
        else:
            spinner = Halo(text="Transcribing...", spinner="dots")
            spinner.start()
            audio = open(audio_file_path, "rb")
            response = self.client.audio.transcriptions.create(
                model="whisper-1", response_format=output_format, file=audio
            )
            transcriptions.append(response)
            spinner.succeed()
        return "\n".join(transcriptions) if len(transcriptions) > 1 else transcriptions[0]


class Utility:
    @staticmethod
    def split_txt_chunks(text, max_chars=500):
        import re

        # Split on sentence boundaries, keeping under max length
        sentences = re.split("(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    @staticmethod
    def split_audio_chunks(audio_file_path: Path, max_size_mb):
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        chunks = []
        current_chunk = AudioSegment.empty()
        current_size = 0

        for i, chunk in enumerate(audio):
            if current_size + len(chunk) > max_size_bytes:
                chunks.append(current_chunk)
                current_chunk = chunk
                current_size = len(chunk)
            else:
                current_chunk += chunk
                current_size += len(chunk)

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    @staticmethod
    def validate_stt_output_format(output_path):
        if not output_path.suffix:
            output_path = output_path.with_suffix(".txt")
            output_format = "text"
        if output_path.suffix == ".txt":
            output_format = "text"
        elif output_path.suffix == ".json":
            output_format = "json"
        elif output_path.suffix == ".vtt":
            output_format = "vtt"
        elif output_path.suffix == ".srt":
            output_format = "srt"
        else:
            raise ValueError("Output file must be a .txt, .json, .vtt, or .srt file")
        return output_path, output_format

    @staticmethod
    def validate_openai_key(api_key):
        if not api_key.startswith("sk-"):
            raise ValueError(
                "Invalid API key. API key must start with 'sk-'. You might have to press `Ctrl + Shift + V` to paste the key."
            )


def parser():
    # Create parser
    parser = argparse.ArgumentParser(
        description="Convert text to speech or speech to text using OpenAI's API"
    )
    sub_parsers = parser.add_subparsers(dest="command")

    # tts subparser
    tts = sub_parsers.add_parser("tts", help="text to speech")
    tts.add_argument("input_file", help="input file")
    tts.add_argument("-k", "--key", help="api key")
    tts.add_argument("-o", "--output", help="output file")
    tts.add_argument(
        "-v",
        "--voice",
        help="voice",
        default="onyx",
        choices=["alloy", "echo", "onyx", "fable", "nova", "shimmer"],
    )
    tts.add_argument(
        "-s",
        "--speed",
        help="The speed of the generated audio. Select a value from `0.25` to `4.0`",
        type=float,
        default=1.0,
    )

    # stt subparser
    stt = sub_parsers.add_parser("stt", help="speech to text using Whisper model")
    stt.add_argument("input_file", help="input file")
    stt.add_argument("-k", "--key", help="api key")
    stt.add_argument("-o", "--output", help="output file")

    # clear cache subparser
    sub_parsers.add_parser("clear", help="clear local cache")

    return parser


def main():
    args = parser().parse_args()

    # If no command is provided, print help
    if not args.command:
        parser().print_help()
        return

    # Only load the config and key if a command other than help is provided
    if args.command == "clear":
        key_path = ConfigManager.get_key_dir()  # clear encryption key
        shutil.rmtree(key_path, ignore_errors=True)
        config_path = ConfigManager.get_config_path()  # clear config file
        if config_path.exists():
            config_path.unlink()
        print("Cache cleared.")
        return

    api_key = args.key
    if not api_key:
        try:
            config = ConfigManager.load_config(encryption=True)
            api_key = config.get("API_KEY", None)
            if not api_key:
                raise KeyError
        except FileNotFoundError:
            user_entered_key = getpass.getpass("Please enter your OpenAI API key: ")
            Utility.validate_openai_key(user_entered_key)
            ConfigManager.save_config({"API_KEY": user_entered_key}, encryption=True)
            api_key = user_entered_key
        except KeyError:
            print("API key not found in config file. Try to run `clear_cache` command.")
    else:
        ConfigManager.save_config({"API_KEY": api_key}, encryption=True)

    openai_handler = OpenAIHandler(api_key=api_key)

    if args.command == "tts":
        with open(args.input_file, encoding="utf-8") as f:
            inputs = f.read()

        output_path = Path(args.output) if args.output else Path("output.mp3")
        if not output_path.suffix:
            output_path = output_path.with_suffix(".mp3")
        if not output_path.suffix == ".mp3":
            raise ValueError("Output file must be an .mp3 file")

        openai_handler.text_to_speech(inputs, output_path, voice=args.voice, speed=args.speed)
        print(f"Audio saved to {output_path}")
    elif args.command == "stt":
        output_path = Path(args.output) if args.output else Path("output.txt")
        output_path, output_format = Utility.validate_stt_output_format(output_path)

        # Transcribe audio
        response = openai_handler.speech_to_text(Path(args.input_file), output_format=output_format)

        # Write response to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Transcription saved to {output_path}")


if __name__ == "__main__":
    main()
