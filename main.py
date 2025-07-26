import speech_recognition as sr
from transformers import pipeline
import ffmpeg
import os
import datetime
import wave
import math


def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def transcribe_long_audio(audio_path, chunk_length=60):
    recognizer = sr.Recognizer()
    transcript = ""
    with sr.AudioFile(audio_path) as source:
        total_duration = int(source.DURATION)
        for i in range(0, total_duration, chunk_length):
            audio = recognizer.record(source, offset=i, duration=chunk_length)
            try:
                text = recognizer.recognize_google(audio)
                transcript += text + " "
            except sr.UnknownValueError:
                print(f"Chunk {i}-{i+chunk_length} sec: Could not understand audio.")
            except sr.RequestError as e:
                print(f"Chunk {i}-{i+chunk_length} sec: API error: {e}")
    return transcript.strip()

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    print("Starting audio extraction...")
    input_folder = 'input/'
    print("Checking for files in input folder...")
    # files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    # Only include supported audio/video files
    supported_exts = ('.mp4', '.mov', '.avi', '.mkv', '.wav', '.flac', '.aiff')
    files = [
    f for f in os.listdir(input_folder)
    if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(supported_exts)]
    if not files:
        print("No files found in input folder.")
        exit(1)
    file_path = os.path.join(input_folder, files[0])
    print(f"Processing file: {file_path}")
    if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            audio_path = extract_audio_from_video(file_path)
    else:
            audio_path = file_path
            
    print(f"Extracted audio to: {audio_path} and file path is {file_path}")

    # Check audio duration
    with wave.open(audio_path, 'rb') as wf:
        duration = wf.getnframes() / float(wf.getframerate())
        print(f"Audio duration: {duration:.2f} seconds")

    print("Transcribing audio...")
    try:
        if duration > 60:
            conversation = transcribe_long_audio(audio_path, chunk_length=60)
        else:
            conversation = transcribe_audio(audio_path)
        print("\nFull Conversation:\n", conversation)
    except FileNotFoundError as e:
        print(f"Error during Transcribing: {e}")
        exit(1)
    except Exception as e:
        print(f"Error during Transcribing: {e}")
        exit(1)

    print("\nSummarizing conversation...")
    try: 
        summary = summarize_text(conversation)
        print("\nSummary:\n", summary)
    except Exception as e:
        print(f"Error during summarization: {e}")
        exit(1)
    
    #create a outputfolder if it does not exist
    output_folder = 'output/'
    #create a folder inside output folder with current date and time to load the output files in txt format
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(output_folder, current_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # #move the audio or video file to the respective output folder
    # output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    # os.rename(file_path, output_file_path)
    # print(f"Moved file to {output_file_path}")
    #save the conversation and summary in a text file
    with open(os.path.join(output_folder, 'conversation.txt'), 'w') as f:
        f.write("Full Conversation:\n" + conversation)
    with open(os.path.join(output_folder, 'summary.txt'), 'w') as f:
        f.write("Summary:\n" + summary)
    print(f"Output saved in {output_folder}")
    print("Process completed successfully.")

