# Install dependencies (uncomment in Colab)
# !pip install -U openai-whisper gradio
# !sudo apt-get install -y ffmpeg

import whisper
import gradio as gr
import tempfile
import os

# Load Whisper model (change "base" → "small", "medium", or "large" for higher accuracy)
model = whisper.load_model("base")

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def transcribe_video(video_file):
    # Save temporary SRT file
    srt_path = tempfile.NamedTemporaryFile(delete=False, suffix=".srt").name
    
    # Transcribe
    result = model.transcribe(video_file, fp16=False)

    # Write SRT file
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"]):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
    
    # Return transcription text + SRT file
    return result["text"], srt_path

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎙 Whisper Video Transcriber")
    gr.Markdown("Upload a video/audio file and get both **text transcription** and **.srt subtitles**.")

    with gr.Row():
        input_file = gr.File(label="Upload Video/Audio", type="filepath")
    
    with gr.Row():
        transcription_output = gr.Textbox(label="Full Transcription", lines=10)
        subtitle_file = gr.File(label="Download Subtitles (.srt)")

    transcribe_btn = gr.Button("Transcribe")
    transcribe_btn.click(transcribe_video, inputs=[input_file], outputs=[transcription_output, subtitle_file])

# Launch app
demo.launch()
