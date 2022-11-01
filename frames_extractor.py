import cv2

class video():
	def __init__(self, video_path, sequence_length=30, min_duration=150):
		self.video_path = video_path
		self.sequence_length = sequence_length
		self.min_duration = min_duration

	def get_no_of_sequences(self, duration):
		'Function to extract frames from a video, as multiples of `sequence_length` and calculated based on the video length'
		if duration <= self.min_duration: return self.sequence_length
		return self.sequence_length + self.get_no_of_sequences(duration - self.min_duration)

	def frames_extraction(self, video_path):
		'''
		This function will extract the required frames from a resized video, and then normalize them.
		Args:
			video_path: The path of the video in the disk, whose frames are to be extracted.
		Returns:
			frames_list: A list containing the resized and normalized frames of the video.
		'''

		# Declare a list to store video frames.
		frames_list = []
		
		# Read the Video File using the VideoCapture object.
		video_reader = cv2.VideoCapture(video_path)

		fps = video_reader.get(cv2.CAP_PROP_FPS)
		frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
		duration = frame_count/fps

		# Take frames every 5 seconds. If video is shorter than 2m:30s, take equally spaced 30 frames.
		sequence_len = self.get_no_of_sequences(duration)

		skip_frames_window = max(int(frame_count/sequence_len), 1)

		# Iterate through the Video Frames.
		for frame_counter in range(sequence_len):
			# Set the current frame position of the video.
			video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

			# Reading the frame from the video. 
			success, frame = video_reader.read() 

			# Check if Video frame is not successfully read then break the loop
			if not success:
				break
			
			# Normalize the frame by dividing it with 255 so that each pixel value then lies between 0 and 1
			normalized_frame = frame / 255
			
			# Append the normalized frame into the frames list
			frames_list.append(normalized_frame)
		
		# Release the VideoCapture object. 
		video_reader.release()

		# Return the frames list.
		return frames_list