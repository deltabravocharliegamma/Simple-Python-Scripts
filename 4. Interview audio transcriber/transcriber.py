from openai import OpenAI
import os
import sys
import tempfile
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import time
import shutil
import asyncio
import aiohttp
from openai import AsyncOpenAI
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()


# Load Trust Store if specified in .env (must install truststore)
if os.getenv("USE_TRUST_STORE_SSL"): #set to true if you want to use while using zscaler
    import truststore
    truststore.inject_into_ssl()
    os.environ.pop("SSL_CERT_FILE", None)


# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your OpenAI API key or set it as an environment variable.")
    sys.exit(1)

def get_ffmpeg_path():
    """Get the path to FFmpeg executable."""
    try:
        if os.name == 'nt':  # Windows
            ffmpeg_path = subprocess.check_output(['where', 'ffmpeg']).decode().strip()
        else:  # Linux and Mac
            ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
        print(f"FFmpeg found at: {ffmpeg_path}")
        return ffmpeg_path
    except subprocess.CalledProcessError:
        print('FFmpeg not found. Please ensure it is installed and added to the PATH.')
        raise FileNotFoundError('FFmpeg not found')

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        # Check for ffmpeg
        get_ffmpeg_path()

        # Check for ffprobe
        if os.name == 'nt':  # Windows
            subprocess.check_output(['where', 'ffprobe'])
        else:  # Linux and Mac
            subprocess.check_output(['which', 'ffprobe'])

        print("All dependencies found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Required dependencies not found. Please ensure ffmpeg and ffprobe are installed.")
        return False

def preprocess_audio(input_file, output_format="ogg", max_size_mb=24):
    """Preprocess audio file for better transcription."""
    try:
        print(f"Starting preprocessing of file: {input_file}")
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
            output_file = temp_file.name

        # Added -y flag to force overwrite without prompting
        command = [
            "ffmpeg",
            "-y",  # Force overwrite without asking
            "-i", input_file,
            "-vn",  # No video
            "-ac", "1",  # Mono audio
            "-ar", "44100",  # 44.1kHz sample rate
            "-q:a", "2",  # Quality setting for audio
            output_file
        ]

        print(f"Executing FFmpeg command: {' '.join(command)}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        if process.returncode != 0:
            print(f"FFmpeg error: {process.stderr}")
            return None

        # Check if the output file is still too large
        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        if output_size_mb > max_size_mb:
            print(f"Processed file still too large ({output_size_mb:.2f} MB). Applying additional compression.")

            # Create another temporary file for the more compressed version
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as compressed_temp:
                compressed_output = compressed_temp.name

            # Apply more aggressive compression
            compress_command = [
                "ffmpeg",
                "-y",
                "-i", output_file,
                "-vn",
                "-ac", "1",
                "-ar", "32000",  # Lower sample rate
                "-b:a", "64k",   # Lower bitrate
                compressed_output
            ]

            print(f"Executing compression command: {' '.join(compress_command)}")
            compress_process = subprocess.run(
                compress_command,
                capture_output=True,
                text=True,
                check=False
            )

            # Delete the first temporary file
            Path(output_file).unlink(missing_ok=True)

            if compress_process.returncode != 0:
                print(f"FFmpeg compression error: {compress_process.stderr}")
                return None

            output_file = compressed_output

        print(f"Preprocessing completed. Output file: {output_file}")
        return output_file
    except Exception as e:
        print(f"Unexpected error during preprocessing: {str(e)}")
        return None

def get_audio_duration(file_path):
    """Get duration of audio file in minutes."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        duration = float(output) / 60  # Convert seconds to minutes
        print(f"Audio duration: {duration:.2f} minutes")
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 0

def get_audio_file_size(file_path):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Audio file size: {size_mb:.2f} MB")
        return size_mb
    except Exception as e:
        print(f"Error getting file size: {str(e)}")
        return 0

def split_long_audio(file_path, max_duration_minutes=25):
    """Split long audio files into chunks of specified maximum duration."""
    try:
        duration = get_audio_duration(file_path)
        if duration <= max_duration_minutes:
            return [file_path]  # No need to split

        print(f"Splitting audio file of {duration:.2f} minutes into {max_duration_minutes}-minute chunks")

        # Create a temporary directory for the chunks
        temp_dir = tempfile.mkdtemp()
        file_extension = os.path.splitext(file_path)[1]
        base_name = os.path.basename(file_path).replace(file_extension, '')

        # Calculate number of chunks needed
        num_chunks = int(duration / max_duration_minutes) + 1

        # Split the audio file
        chunk_files = []
        for i in range(num_chunks):
            start_time = i * max_duration_minutes * 60  # Convert to seconds
            output_file = os.path.join(temp_dir, f"{base_name}_part{i+1}{file_extension}")

            command = [
                "ffmpeg",
                "-y",
                "-i", file_path,
                "-ss", str(start_time),
                "-t", str(max_duration_minutes * 60),  # Duration in seconds
                "-c", "copy",  # Copy without re-encoding
                output_file
            ]

            print(f"Executing split command for chunk {i+1}/{num_chunks}")
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )

            if process.returncode != 0:
                print(f"Error splitting chunk {i+1}: {process.stderr}")
                continue

            chunk_files.append(output_file)

        print(f"Split audio into {len(chunk_files)} chunks")
        return chunk_files
    except Exception as e:
        print(f"Error splitting audio file: {str(e)}")
        return [file_path]  # Return original file on error

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def transcribe_with_whisper(file_path, model="whisper-1", max_retries=3):
    """Transcribe an audio file using OpenAI's Whisper model."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        print(f"Starting transcription of file: {file_path}")

        # Check file size and duration
        file_size = get_audio_file_size(file_path)
        duration = get_audio_duration(file_path)

        # Process file if needed (OpenAI has a 25MB limit)
        processed_file = file_path
        temp_files_to_delete = []

        # If file is too long, split it
        if duration > 25:
            print(f"Audio duration ({duration:.2f} min) exceeds limit. Splitting into chunks.")
            chunk_files = split_long_audio(file_path)

            if len(chunk_files) > 1:
                # Transcribe each chunk and combine
                all_transcriptions = []

                # Process chunks concurrently with semaphore for rate limiting
                semaphore = asyncio.Semaphore(4)  # Limit concurrent API calls

                async def process_chunk(chunk_file, chunk_index):
                    async with semaphore:
                        print(f"Transcribing chunk {chunk_index+1}/{len(chunk_files)}")

                        # Check if chunk needs preprocessing
                        chunk_size = get_audio_file_size(chunk_file)
                        if chunk_size > 24:
                            processed_chunk = preprocess_audio(chunk_file)
                            if not processed_chunk:
                                return f"[Failed to preprocess chunk {chunk_index+1}]"
                            temp_files_to_delete.append(processed_chunk)
                        else:
                            processed_chunk = chunk_file

                        # Transcribe with retries
                        for attempt in range(max_retries):
                            try:
                                with open(processed_chunk, "rb") as audio_file:
                                    chunk_transcription = await client.audio.transcriptions.create(
                                        model=model,
                                        file=audio_file,
                                        response_format="text"
                                    )
                                return chunk_transcription
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                                    await asyncio.sleep(5)
                                else:
                                    return f"[Transcription failed for part {chunk_index+1}]"

                # Process all chunks concurrently
                tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunk_files)]
                all_transcriptions = await asyncio.gather(*tasks)

                # Clean up temporary chunk files
                for chunk_file in chunk_files:
                    if chunk_file != file_path:  # Don't delete the original file
                        Path(chunk_file).unlink(missing_ok=True)

                # Clean up the temp directory
                if len(chunk_files) > 1 and os.path.dirname(chunk_files[0]) != os.path.dirname(file_path):
                    shutil.rmtree(os.path.dirname(chunk_files[0]), ignore_errors=True)

                # Combine all transcriptions
                transcription = "\n\n".join(all_transcriptions)

                # Clean up any other temp files
                for temp_file in temp_files_to_delete:
                    Path(temp_file).unlink(missing_ok=True)

                print("Combined transcription of all chunks completed successfully")
                return transcription

        # If file is too large but not too long, preprocess it
        if file_size > 24:
            print(f"File size: {file_size:.2f} MB exceeds limit. Preprocessing required.")
            processed_file = preprocess_audio(file_path)
            if not processed_file:
                raise Exception("Audio preprocessing failed")
            temp_files_to_delete.append(processed_file)

        # Transcribe with retries
        for attempt in range(max_retries):
            try:
                # Transcribe the audio
                with open(processed_file, "rb") as audio_file:
                    transcription = await client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="text"
                    )

                # Clean up temporary files
                for temp_file in temp_files_to_delete:
                    Path(temp_file).unlink(missing_ok=True)

                print("Transcription completed successfully")
                return transcription
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    # Clean up temporary files on final failure
                    for temp_file in temp_files_to_delete:
                        Path(temp_file).unlink(missing_ok=True)
                    print(f"All {max_retries} attempts failed: {str(e)}")
                    return f"Transcription error: {str(e)}"
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Transcription error: {str(e)}"

async def summarize_transcript(transcript, model="gpt-4o-mini", max_retries=3):
    """Summarize the transcript using OpenAI's chat completions."""
    try:
        if not transcript or transcript.startswith("Transcription error"):
            return "Cannot summarize due to transcription error."

        print("Starting transcript summarization")

        prompt_option_1 = """You are a highly skilled assistant specialized in analyzing and summarizing interview transcripts.
        Please provide a comprehensive summary that includes:
        1. Main topics and key points discussed
        2. Important insights or opinions expressed
        3. Notable quotes (with attribution if possible)
        4. Key themes that emerged in the interview
        
        Format the summary with clear sections and bullet points for readability."""

        prompt_option_2 = """This GPT is a business professional assistant specializing in analyzing meeting transcripts. Its main goal is to identify the type of meeting (such as steering committee readout, internal team meeting, client meeting, 1-on-1, topic deep dive, expert interview, etc.) and produce a structured summary. The summary is divided into clear sections: 'Meeting Type / Objectives', 'Overall Topic', 'Attendees', 'Discussion Notes', 'Key Takeaways,' 'Next Steps (with Owners)', and, when appropriate, 'Memorable Quotes'.
        It is designed to streamline meeting follow-ups, ensure clarity, and surface actionable insights. It places strong emphasis on clearly documenting next steps along with ownership and on capturing any key decisions made during the meeting.
        When the transcript suggests an interview or strong opinions were shared, it adds a 'Memorable Quotes" section. If the speaker attribution is unclear, it includes the quote with a note of uncertainty about the speaker. It assumes input will be a raw or unstructured transcript and makes reasoned assumptions when data is missing.
        It infers meeting types based on conversational cues, participant roles, and agenda items. The assistant always uses concise business language, keeps the tone professional, and avoids speculation beyond business-reasonable assumptions. It asks clarifying questions only when necessary. The format and structure are consistent across outputs to ensure readability and ease of downstream sharing, storage, or documentation.
        This assistant is ideal for consultants, analysts, executives, and project managers who need fast, reliable meeting recaps and insight distillation."""

        # Get prompt from env if exists
        env_prompt = os.getenv("CUSTOM_PROMPT")

        # Use custom prompt if in .env file
        system_prompt = env_prompt if env_prompt else prompt_option_1

        # Try with retries
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please analyze and summarize this interview transcript:\n\n{transcript}"}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )

                summary = response.choices[0].message.content
                print("Summary generated successfully")
                return summary
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print(f"All {max_retries} attempts failed: {str(e)}")
                    return f"Summarization error: {str(e)}"
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return f"Summarization error: {str(e)}"

async def process_audio_file(file_path, output_dir=None, transcription_model="whisper-1",
                             summary_model="gpt-4o-mini", create_summary=True,
                             client_name=None, meeting_name=None, original_video_path=None):
    """Process a single audio file: transcribe and optionally summarize, then organize if enabled."""
    try:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        if output_dir is None:
            script_dir = Path(__file__).parent
            output_dir = script_dir / "Output files"

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing file: {file_name}")

        # Transcribe
        transcript = await transcribe_with_whisper(file_path, transcription_model)
        if transcript.startswith("Transcription error"):
            print(f"Transcription failed for {file_name}: {transcript}")
            error_path = os.path.join(output_dir, f"{base_name}_error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error processing {file_name}:\n{transcript}")
            return False

        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved to {transcript_path}")

        summary_path = None
        combined_path = os.path.join(output_dir, f"{base_name}_full.txt")

        if create_summary:
            summary = await summarize_transcript(transcript, summary_model)
            if summary.startswith("Summarization error"):
                print(f"Summarization failed for {file_name}: {summary}")
                with open(combined_path, "w", encoding="utf-8") as f:
                    f.write("TRANSCRIPT\n" + "="*50 + "\n" + transcript)
                    f.write("\n\nSUMMARY\n" + "="*50 + "\n")
                    f.write(f"Summary generation failed: {summary}")
            else:
                summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"Summary saved to {summary_path}")

                with open(combined_path, "w", encoding="utf-8") as f:
                    f.write("TRANSCRIPT\n" + "="*50 + "\n" + transcript)
                    f.write("\n\nSUMMARY\n" + "="*50 + "\n" + summary)
                print(f"Combined transcript and summary saved to {combined_path}")

        # Organize if applicable
        if client_name and meeting_name:
            try:
                if not os.path.exists(file_path):
                    print(f"File already moved — skipping reorganization: {file_path}")
                else:
                    organize_files(
                        client_name=client_name,
                        meeting_name=meeting_name,
                        audio_file_path=file_path,
                        transcript_file=transcript_path,
                        summary_file=summary_path if create_summary else None,
                        combined_file=combined_path,
                        original_video_path=original_video_path
                    )
            except Exception as e:
                # Silently skip organizing errors (already moved or permission denied)
                pass

        return True

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            error_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error processing {os.path.basename(file_path)}:\n{str(e)}")
        except:
            pass
        return False


def clean_env_path(path):
    """Cleans and normalizes a .env path with quotes."""
    if not path:
        return ""
    # Strip surrounding quotes and replace escape sequences
    return os.path.normpath(path.strip().strip('"').replace('\\', '/'))

def get_user_input():
    """Get user input for processing parameters."""
    print("\nAudio Transcription and Summarization Tool")
    print("=========================================")

    script_dir = Path(__file__).parent
    default_input_dir = script_dir / "Input files"
    default_output_dir = script_dir / "Output files"

    audio_env = clean_env_path(os.getenv("DEFAULT_AUDIO_FILE_LOCATION"))
    output_env = clean_env_path(os.getenv("DEFAULT_OUTPUT_LOCATION"))

    if audio_env:
        default_input_dir = audio_env.strip().strip('"')
    if output_env:
        default_output_dir = output_env.strip().strip('"')

    input_prompt = f"\nEnter the path to the folder containing audio files (default: {default_input_dir}): "
    input_dir = input(input_prompt).strip()
    if not input_dir:
        input_dir = default_input_dir
    else:
        # Convert to string before using string methods
        input_dir = str(input_dir).replace('"', '').replace("'", "")  # Remove quotes if present

    output_prompt = f"\nEnter the path to save output files (default: {default_output_dir}): "
    output_dir = input(output_prompt).strip()
    if not output_dir:
        output_dir = default_output_dir
    else:
        # Convert to string before using string methods
        output_dir = str(output_dir).replace('"', '').replace("'", "")  # Remove quotes if present

    # Get transcription model
    transcription_models = ["whisper-1"]
    print("\nAvailable transcription models:")
    for i, model in enumerate(transcription_models, 1):
        print(f"{i}. {model}")

    transcription_model = transcription_models[0]  # Default to first model

    # Get summary model
    summary_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    print("\nAvailable summary models:")
    for i, model in enumerate(summary_models, 1):
        print(f"{i}. {model}")
    while True:
        model_choice = input(f"Select a summary model (1-{len(summary_models)}, default is 1): ").strip()
        if not model_choice:
            summary_model = summary_models[0]
            break
        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(summary_models):
                summary_model = summary_models[model_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(summary_models)}.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        create_summary = input("Generate summaries for transcripts? (y/n, default is y): ").strip().lower()
        if not create_summary or create_summary == 'y':
            create_summary = True
            break
        elif create_summary == 'n':
            create_summary = False
            break
        else:
            print("Please enter 'y' or 'n'.")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "transcription_model": transcription_model,
        "summary_model": summary_model,
        "create_summary": create_summary
    }


def convert_mkv_to_wav(input_dir):
    """
    Converts all .mkv files in the input directory to .wav format using ffmpeg.
    Returns a mapping of .wav file → original .mkv file.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return {}

    mkv_files = list(input_path.glob("*.mkv"))
    if not mkv_files:
        return {}

    print(f"Found {len(mkv_files)} .mkv file(s) to convert to .wav")
    wav_to_mkv = {}

    for video_file in mkv_files:
        audio_file = video_file.with_suffix(".wav")
        if audio_file.exists():
            print(f".wav already exists for: {video_file.name}")
        else:
            command = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                str(audio_file)
            ]
            print(f"Extracting audio from: {video_file.name}")
            try:
                subprocess.run(command, check=True, capture_output=True)
                print(f"Created: {audio_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to convert {video_file.name}: {e.stderr.decode().strip()}")
                continue

        wav_to_mkv[str(audio_file)] = str(video_file)

    return wav_to_mkv

async def process_audio_folder(input_dir, output_dir=None, transcription_model="whisper-1", summary_model="gpt-4o-mini", create_summary=True):
    """Process all audio files in a folder, collecting metadata first if organizing is enabled."""
    try:
        if not os.path.isdir(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            return False

        if output_dir is None:
            output_dir = os.path.join(input_dir, "transcripts")

        os.makedirs(output_dir, exist_ok=True)

        # Convert .mkv files and get mapping
        wav_to_mkv_map = convert_mkv_to_wav(input_dir)

        audio_extensions = ('.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma')
        audio_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(audio_extensions)
        ]

        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return False

        audio_files.sort()
        print(f"\nFound {len(audio_files)} audio files to process:")
        for i, file_path in enumerate(audio_files, 1):
            print(f"  {i}. {os.path.basename(file_path)}")

        organize_flag = os.getenv("ORGANIZE_FILES", "false").strip().lower() == "true"

        metadata_list = []
        for file_path in audio_files:
            client_name = None
            meeting_name = None
            original_video_path = wav_to_mkv_map.get(file_path)

            if organize_flag:
                print(f"\nMetadata for: {os.path.basename(file_path)}")
                client_name = input("  → Enter client name: ").strip()
                while not client_name:
                    print("    Client name cannot be empty.")
                    client_name = input("  → Enter client name: ").strip()

                meeting_name = input("  → Enter meeting name: ").strip()
                while not meeting_name:
                    print("    Meeting name cannot be empty.")
                    meeting_name = input("  → Enter meeting name: ").strip()

            metadata_list.append({
                "file_path": file_path,
                "client_name": client_name,
                "meeting_name": meeting_name,
                "original_video_path": original_video_path
            })

        semaphore = asyncio.Semaphore(4)

        async def process_with_metadata(meta):
            async with semaphore:
                return await process_audio_file(
                    file_path=meta["file_path"],
                    output_dir=output_dir,
                    transcription_model=transcription_model,
                    summary_model=summary_model,
                    create_summary=create_summary,
                    client_name=meta["client_name"],
                    meeting_name=meta["meeting_name"],
                    original_video_path=meta.get("original_video_path")
                )

        tasks = [process_with_metadata(meta) for meta in metadata_list]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        failed = len(results) - successful

        report_path = os.path.join(output_dir, "transcription_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Transcription Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total files processed: {len(audio_files)}\n")
            f.write(f"Successfully processed: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            f.write("Files processed:\n")
            for fp in audio_files:
                f.write(f"- {os.path.basename(fp)}\n")

        print(f"Processing complete. Successful: {successful}, Failed: {failed}")
        print(f"Report saved to {report_path}")
        return True

    except Exception as e:
        print(f"Error processing folder {input_dir}: {str(e)}")
        return False


async def async_main():
    """Async main function to run the script."""
    # Check for dependencies
    if not check_dependencies():
        print("Required dependencies not found. Please install ffmpeg and ffprobe.")
        sys.exit(1)

    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(description='Transcribe audio files and generate summaries.')
        parser.add_argument('input_dir', help='Path to the folder containing audio files')
        parser.add_argument('--output', help='Path to save the output files', default=None)
        parser.add_argument('--transcription-model', help='OpenAI model to use for transcription', default="whisper-1")
        parser.add_argument('--summary-model', help='OpenAI model to use for summarization', default="gpt-4o-mini")
        parser.add_argument('--no-summary', help='Skip generating summaries', action='store_true')
        parser.add_argument('--single-file', help='Process a single file instead of a directory', action='store_true')

        args = parser.parse_args()
        input_dir = args.input_dir
        output_dir = args.output
        transcription_model = args.transcription_model
        summary_model = args.summary_model
        create_summary = not args.no_summary

        # Handle single file mode
        if args.single_file:
            if not os.path.isfile(input_dir):
                print(f"Error: '{input_dir}' is not a valid file.")
                sys.exit(1)

            if output_dir is None:
                # Use default output directory
                script_dir = Path(__file__).parent
                output_dir = script_dir / "Output files"

            success = await process_audio_file(
                input_dir,
                output_dir,
                transcription_model,
                summary_model,
                create_summary
            )

            if success:
                print(f"\nProcessing complete! Results saved to {output_dir}")
            else:
                print("\nProcessing failed. Check the log file for details.")

            sys.exit(0)
    else:
        # Get input interactively
        user_input = get_user_input()
        input_dir = user_input["input_dir"]
        output_dir = user_input["output_dir"]
        transcription_model = user_input["transcription_model"]
        summary_model = user_input["summary_model"]
        create_summary = user_input["create_summary"]

    # Process the folder
    success = await process_audio_folder(
        input_dir,
        output_dir,
        transcription_model,
        summary_model,
        create_summary
    )

    if success:
        print(f"\nProcessing complete! Results saved to {output_dir}")
    else:
        print("\nProcessing failed. Check the log file for details.")
def sanitize_name(name: str) -> str:
    """Clean up user input for safe filenames and folder names."""
    forbidden_chars = r'<>:"/\\|?*&'
    clean = name.strip()
    for char in forbidden_chars:
        clean = clean.replace(char, '-')
    return clean.replace("  ", " ")

def organize_files(client_name: str, meeting_name: str, audio_file_path: str,
                   transcript_file: str, summary_file: str, combined_file: str,
                   original_video_path: str = None):
    # Check if organizing is enabled
    organize_flag = os.getenv("ORGANIZE_FILES", "false").strip().lower()
    if organize_flag != "true":
        print("File organization is disabled. Files will remain in the output directory.")
        return False

    parent_dir = os.getenv("DEFAULT_ORGANIZED_FOLDER_LOCATION")
    if not parent_dir:
        print("DEFAULT_ORGANIZED_FOLDER_LOCATION not defined in .env")
        return False

    file_date = get_file_date(audio_file_path)
    client_name_clean = sanitize_name(client_name)
    meeting_name_clean = sanitize_name(meeting_name)
    base_filename = f"{file_date} {client_name_clean} {meeting_name_clean}"

    client_folder = Path(parent_dir) / client_name_clean
    client_folder.mkdir(parents=True, exist_ok=True)

    # New paths
    new_audio_path = client_folder / f"{base_filename}{Path(audio_file_path).suffix}"
    new_transcript_path = client_folder / f"{base_filename}_transcript.txt"
    new_combined_path = client_folder / f"{base_filename}_full.txt"

    # Move and rename .wav
    if os.path.exists(audio_file_path):
        new_audio_path = resolve_conflict(new_audio_path)
        shutil.move(audio_file_path, new_audio_path)
    else:
        print(f"Skipped moving audio file (already moved?): {audio_file_path}")

    # Move and rename .mkv
    if original_video_path and os.path.exists(original_video_path):
        new_video_path = client_folder / f"{base_filename}{Path(original_video_path).suffix}"
        new_video_path = resolve_conflict(new_video_path)
        shutil.move(original_video_path, new_video_path)
    else:
        if original_video_path:
            print(f"Skipped moving video file (already moved?): {original_video_path}")

    # Move outputs
    new_transcript_path = resolve_conflict(new_transcript_path)
    shutil.move(transcript_file, new_transcript_path)
    new_combined_path = resolve_conflict(new_combined_path)
    shutil.move(combined_file, new_combined_path)

    if summary_file and os.path.exists(summary_file):
        new_summary_path = client_folder / f"{base_filename}_summary.txt"
        new_summary_path = resolve_conflict(new_summary_path)
        shutil.move(summary_file, new_summary_path)

    print(f"Files organized in: {client_folder}")
    return True

def get_file_date(filepath):
    # Get last modified timestamp
    timestamp = os.path.getmtime(filepath)
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

def resolve_conflict(path: Path) -> Path:
    counter = 1
    base = path.stem
    suffix = path.suffix
    parent = path.parent

    while path.exists():
        path = parent / f"{base} ({counter}){suffix}"
        counter += 1

    return path


if __name__ == "__main__":
    asyncio.run(async_main())
