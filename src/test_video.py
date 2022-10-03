import pandas as pd

from torchvision.io import read_video
import cv2


if __name__ == "__main__":
    df = pd.read_csv("data_uids/training.csv")
    # print (df.head())
    ix= 80
    file_name = "v1/clips/" + df.iloc[ix]['clip_uid'] + '.mp4'
    print (file_name)
    id = df.iloc[ix]

    start_time =246.80284
    print (start_time)
    end_time = 480.022
    print(end_time)
    torch_read = read_video(file_name,start_pts=start_time,end_pts=end_time,pts_unit='sec')
    print (torch_read)
    video = torch_read[0].numpy()
    print (video.shape)
    frame_counter = 0
    while frame_counter<torch_read[0].size(0):
        try:
            cv2.imshow("frame",video[frame_counter])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except:
            break
        frame_counter+=1


    # print(torch_read[0].size())
    # read_video = cv2.VideoCapture(file_name)
    # video = torch_read[0].numpy()
    # print (video.size())
    


    
    
    
    frame_counter = 0




    # while read_video.isOpened():
    #     ret,frame = read_video.read()
    #     if ret==True:
    #         frame = cv2.resize(frame,(500,500))
    #         cv2.imshow("Frame",frame)

    #         print(str(read_video.get(cv2.CAP_PROP_POS_MSEC)/60))
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #     frame_counter+=1
    #     print (frame_counter)
    # frame_video = read_video(file_name,start_pts=video_startframe,end_pts=video_endframe,pts_unit='sec',output_format='TCHW')
