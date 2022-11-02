import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input

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
		for i, ID in enumerate(list_IDs_temp):
			frame_data = np.load(ID)
			# pre-process the frames for VGG16 input
			frame_data = frame_data/127.5
			frame_data -= 1.
			X.append(frame_data)
			# Store class
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