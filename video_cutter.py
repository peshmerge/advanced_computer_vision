import datetime

from moviepy.editor import *


def cut_video(video_path, start_time,end_time,save_path):
    # Convert the amount of seconds in time xx:xx:xx.xxxxxx
    start_time = str(datetime.timedelta(seconds = start_time))
    end_time = str(datetime.timedelta(seconds = end_time))
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)
    clip.write_videofile(save_path)
# Example
cut_video("1.mp4",1486.1536052666665,1508.2152952666665,"111.mp4")
