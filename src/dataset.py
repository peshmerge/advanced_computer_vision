import json
import pprint
import cv2
from collections import OrderedDict
import pandas as pd
# from experimental_setup import train
import torchvision


import numpy as np 
import torch
from torchvision.transforms import Resize
from torchvision.io import read_video
from sklearn.model_selection import train_test_split


from torchvideotransforms import video_transforms, volume_transforms
import os




"""
Takes the processed csv file as input 

"""


class Ego4D:
    """
    

    
    => transform :- transformation function
    => playOnIndexing :- If set to true, on indexing the video will play.(turn it off when training the model)
    => min_frames :- Number of frames you want
    => offset_values :- offset for the min_frames












    => use_preprocessed_df:- It is set to false if you want to explore the whole dataset, if set to true
    it will use the filtered version based on the value you set for the variable 'min_frames' and 'offset_values'

    
    """

    def __init__(self,transform,df,playOnIndexing=True,seq_length = 30,filter_dataframe = False,clipPath=""):

        self.seq_length = seq_length
        self.df = pd.read_csv(df) if not filter_dataframe else self.filter_dataframe(pd.read_csv(df))
        self.printStats()
        self.playOnIndexing = playOnIndexing
        self.transform = transform
       
        self.classes = self.df['label'].unique() 
        print(self.classes)
        self.class_ix = {v:k for k,v in enumerate(self.classes)} #Create mapping for class to index

        self.ix_class = {k:v for k,v in enumerate(self.classes)} #Create mapping for index to class
        self.clipPath = clipPath

    def __getitem__(self,ix):
        file_name = self.clipPath + self.df.iloc[ix]['clip_uid']+'.mp4'

        label = self.df.iloc[ix]['label']
        video_startframe = self.df.iloc[ix]['start_frame_sec']
        video_endframe = self.df.iloc[ix]['end_frame_sec']

        frame_video = read_video(file_name,start_pts=video_startframe,end_pts=video_endframe,pts_unit='sec',output_format='TCHW')
        #Slices the first min frames
        getFrames = frame_video[0]
        if self.transform:
            # getFrames = torch.permute(getFrames,(0,3,1,2))
            getFrames = self.transform(getFrames)
        
        
        # print (getFrames.size())


        if self.playOnIndexing:
            print (f"Label : {label}")

            frame_count = 0
            new = np.transpose(getFrames.numpy(),(0,2,3,1))


            # print (new.shape)
            while frame_count<=new.shape[0]:
                try:
                    
                    cv2.imshow(label,new[frame_count])
                    frame_count+=1
                except IndexError:
                    break
                if cv2.waitKey(10) ==ord('q'):
                    break
                
        
        total_frames = getFrames.size(0)
     
        if self.seq_length<total_frames:
            getFrames = getFrames[:self.seq_length]
        else:
            getFrames = self.pad_sequence(getFrames,self.seq_length)
        return getFrames.type(torch.FloatTensor),self.class_ix[label]


    def __len__(self):
        return len(self.df)
    

    def pad_sequence(self,frame,seq_length): #pad frame with zeros
        new_frame = torch.zeros((seq_length,frame.size(1),
                            frame.size(2),
                            frame.size(3)))
        # print(frame.size())
        new_frame[seq_length-frame.size(0):] = frame
        return new_frame



    """
        To create the training and testing set

        Parameters
        =======================
        
        sample_each_class :- Number of samples for each class 
        classes :- Array which contains the list of classes you want to take a subset of
        filename :- This function will save two files
        (a) A dataframe containing all the details such as, filename, start_sec, end_sec
        (b) A text file containing the clip uids so it is easy downloadable

        So you can create a training and testing set
    """


 

    def filter_dataframe(self,df):

        """
            This function fetches the video, and slices
            the video starting from the number of frames based thats given.
            



        """


        """

        The line below filters out the dataframe consisting of total frames>=slice frames and
        adding a conditional statement by adding an offset value, so we get slice_frames+offset_value. This is to avoid taking videos
        with high frames, because for example if we are taking 30 frames, and a video consists of 1200 frames, taking 30 frames out of the 1200 is a loss of information.
        So we take the videos which have frames > than the slice_frame variable, and total_frames<= slice_frames+offset_value
        """

        return df.loc[(df['total_frames']>=self.min_frames) & (df['total_frames']<=self.min_frames+self.offset_value)]
        
        
    def printStats(self):

        print (f"Total number of classes : {len(self.df['label'].unique())}")
        print (f"Min frames : {min(self.df['total_frames'])}")
        print (f"Max frame : {max(self.df['total_frames'])}")
        print ("All Labels")
        print ('='*50)
        print (self.df['label'].value_counts())
        # print ("="*50)
        # print (self.df['video_uid'].unique())

# print ("sdasdasasd")


# b = torch.randn(40,3,500,500)
# c = Resize((224,224))
# print(c(b).size())

# dataset= Ego4D(transform=None,df="genvideo.csv",seq_length=30,clipPath="v1/clips/")

# print (dataset[100])

# dataset = Ego4D(transform=None,df="training.csv",playOnIndexing=True,min_frames=30,
#             offset_value=20, filter_dataframe=True,clipPath="v1/clips/")  

# dataset[1]
    
# #Create the training set from the pre-processed, containing the filtered version of the dataframe
# dataset.create_set(sample_each_class=0.2,
#                   classes=['use_phone','clean','converse','read_a_book','throw_away_trash'],
#                   training_file="training.csv",
#                   testing_file="testing.csv")


