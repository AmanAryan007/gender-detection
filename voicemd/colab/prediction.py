import os
import shutil
from pytube import YouTube
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip

from voicemd.predict import make_a_prediction
from voicemd.colab.clean_uploads import re_arrange_files

# Function to download audio from YouTube video
def download_audio_from_youtube(url, output_path):
    yt = YouTube(url)
    video_stream = yt.streams.filter(file_extension="mp4").first()
    video_stream.download(output_path)

    # Convert video to audio using moviepy
    video_clip = VideoFileClip(os.path.join(output_path, video_stream.default_filename))
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(os.path.join(output_path, 'audio_from_video.wav'))

    # Return the name of the audio file
    return 'audio_from_video.wav'

# Clean up existing files
re_arrange_files()

# YouTube URL
youtube_url = input("enter youtube url")

# Download audio from YouTube
downloaded_file = download_audio_from_youtube(youtube_url, './audio_files/')

# Make a prediction using the downloaded audio file
make_a_prediction('./audio_files/' + downloaded_file,
                  config_filepath='./gender-detection/voicemd/config.yaml',
                  best_model_path='./gender-detection/model.pt')

# Clean up temporary files
shutil.rmtree('./audio_files/')
