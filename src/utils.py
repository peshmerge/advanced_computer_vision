import cv2
import json
import pprint
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# from experimental_setup import validation

"""
Gets the device CUDA/CPU

"""

def checkCuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device






"""
Creates a processed csv based version of the moments annotation files, with the required number of columns(in my opinion)
Basically parses the annotation file.

Parameters:- 
momentsAnnotationFiles :- location of where the moments annotation is located
videoPath:- location of where the full_scale videos are. (not needed because we using clips)

"""



def create(momentsAnnotationFile = r"moments_val.json",
videoPath="v1/full_scale/",saveFileName="annot.csv"):
    """
        To create the dataset. 
        This function will iterate through each video, take portions
        of the video as specified in the annotation files.
        A CSV file will be created. 

    """
    df = pd.DataFrame(columns={"label","filename","start_frame","end_frame","start_frame_sec","end_frame_sec","total_frames","clip_uid","video_uid"})
    annotations = json.load(open(momentsAnnotationFile))
    for clips in annotations['videos']:
        for k in clips['clips']:
            for l in k['annotations']:
                for m in l['labels']:
                    df.loc[len(df)+1,['label','filename','start_frame','end_frame','start_frame_sec','end_frame_sec','total_frames',"clip_uid","video_uid"]] = [
                        m['label'].split('_/_')[0],videoPath + clips['video_uid']+'.mp4',
                        m['video_start_frame'],m['video_end_frame'],m['start_time'],m['end_time'],
                        abs(m['video_end_frame']-m['video_start_frame']),k["clip_uid"],clips['video_uid']]


    df.to_csv(saveFileName)


"""
Create training and testing set given the dataframe.

"""

def create_set(df_path,sample_each_class=0.1,
                  classes=['use_phone','clean','read_a_book'],
                  training_file = "training_annot.csv",
                  testing_file = "testing_annot.csv",min_frames=60):

    df = pd.read_csv(df_path)
    training_set = pd.DataFrame(columns={"label","filename","start_frame","end_frame","start_frame_sec","end_frame_sec","total_frames","clip_uid","video_uid"})
    validation_set = pd.DataFrame(columns={"label","filename","start_frame","end_frame","start_frame_sec","end_frame_sec","total_frames","clip_uid","video_uid"})
    for _class in classes:
        getRecords = df.loc[df['label']==_class]
        train,test = train_test_split(getRecords,test_size=sample_each_class)
        training_set = pd.concat([training_set,train])
        validation_set = pd.concat([validation_set,test])
    

    if min_frames:
        training_set = training_set[training_set['total_frames']<=min_frames]
        validation_set = validation_set.loc[validation_set['total_frames']<=min_frames]

    training_set.to_csv(training_file)
    validation_set.to_csv(testing_file)




    with open(f'{training_file}_uids.txt', 'w') as f:
        for i in training_set['clip_uid'].unique():
            # print (i)
            f.write(i)
        
            f.write(" ")

    with open(f'{testing_file}_uids.txt', 'w') as f:
        for i in validation_set['clip_uid'].unique():
            # print (i)
            f.write(i)
        
            f.write(" ")      


    print("Class proportion")
    for i in classes:
        training_class = len(training_set.loc[training_set['label']==i])
        testing_class = len(validation_set.loc[validation_set['label']==i])
        print(f"For class {i}")
        print('='*30)
        print(f"Training : {training_class}")
        print(f"Validation : {testing_class}")

    print("[+]Saved set.")


"""
Make CSV based on the current video files we have
@Input : Takes in the preprocessed csv from the create() function.

"""

def generateCurrentVideos(df_path = "full_annotations.csv",video_path =  "v1/clips/"):
    getVideoList = os.listdir(video_path)
    read_df = pd.read_csv(df_path)
    new_df = pd.DataFrame (columns={"label","filename","start_frame","end_frame","start_frame_sec","end_frame_sec","total_frames","clip_uid","video_uid"})
    
    for i in getVideoList:
        i = i[:-4] #To avoid the mp4 files
        filterDfByVideo = read_df.loc[read_df['clip_uid']==i]
        new_df = pd.concat([new_df,filterDfByVideo])
    new_df.to_csv("genvideo.csv")



if __name__ =="__main__":
    create_set(df_path="genvideo.csv",min_frames=40)
    # generateCurrentVideos()