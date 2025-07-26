import whisper
from transformers import pipeline
import ffmpeg
import os
import datetime
import wave
import keras_nlp
from llama_cpp import Llama
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    return audio_path

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    # summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # device=-1 means use CPU

    # prompt = "Summarize the following text in 5 sentences:\n" + text

    # summary = summarizer(prompt, max_length=120, min_length=30, do_sample=False)
    # print(summary[0]['summary_text'])
    model = keras_nlp.models.BartSeq2SeqLM.from_preset("bart_large_en_cnn")
    summary = model.generate(text, max_length=150 )
    return summary

if __name__ == "__main__":
    print("Starting audio extraction...")
    input_folder = 'input/'
    print("Checking for files in input folder...")
    supported_exts = ('.mp4', '.mov', '.avi', '.mkv', '.wav', '.flac', '.aiff')
    files = [
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(supported_exts)
    ]
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

    print("Transcribing audio with Whisper...")
    try:
        conversation = transcribe_audio_whisper(audio_path)
        print("\nFull Conversation:\n", conversation)
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
    
    # Create output folder if it does not exist
    output_folder = 'output/'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(output_folder, current_time)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the conversation and summary in a text file
    with open(os.path.join(output_folder, 'conversation.txt'), 'w') as f:
        f.write("Full Conversation:\n" + conversation)
    with open(os.path.join(output_folder, 'summary.txt'), 'w') as f:
        f.write("Summary:\n" + summary)
    print(f"Output saved in {output_folder}")
    print("Process completed successfully.")
