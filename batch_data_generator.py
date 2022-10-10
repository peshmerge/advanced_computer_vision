import numpy as np
import tensorflow as tf
import cv2

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=250, sequence_length=30, n_classes=7, min_duration=150, shuffle=True):
		'''
		Generates data for Keras for our ego4D dataset.
		Args:
			list_IDs: A list containing the paths of the videos.
			labels: A list containing the labels of the corresponding videos.
			batch_size: The batch size to be used (# of videos per batch).
			sequence_length: The length of a sequence to be fed to the model.
			n_classes: The number of classes in the dataset.
			min_duration: The minimum duration in seconds to extract sequence in multiples of.
			shuffle: Whether to shuffle the data after each epoch.
		'''
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.labels = labels
		self.sequence_length = sequence_length
		self.min_duration = min_duration
		self.n_classes = n_classes

		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = []
		y = []

		# Generate data
		for idx, ID in enumerate(list_IDs_temp):
			frames = self.frames_extraction(ID)
			# Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
			# Append the data to their repective lists.
			for i in range(0, len(frames), self.sequence_length):
				X.append(frames[i:i+self.sequence_length])
				y.append(self.labels[ID])

		return np.asarray(X), tf.keras.utils.to_categorical(np.array(y), num_classes=self.n_classes)

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

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