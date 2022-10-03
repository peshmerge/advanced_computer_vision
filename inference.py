
"""
To test the model, while the video is
playing.
"""
from torch import nn
import torch
from torchvision.io import read_video
from torchvideotransforms import video_transforms, volume_transforms
import os
import cv2
from src.utils import checkCuda
from torchvision.transforms import Resize
from torchvision import transforms

def load_model(model_path="saved-model/"):

    return torch.load(model_path,map_location=torch.device('cuda'))

def predict(video_path,model):
    
    video = read_video(video_path)[0].numpy()
    # print (video.shape)
    video_transform_list = [video_transforms.Resize((224,224)),
                        volume_transforms.ClipToTensor()]
    transforms = video_transforms.Compose(video_transform_list)

    torch_video = transforms(video)

    print (torch_video.size())
    numpy_video = torch.permute(torch_video,(1,2,3,0)).numpy()
    print (torch_video.shape)
    print (numpy_video.shape)

    prev_counter = 0
    frame_counter = 0
    input = None
    prediction = "ideal"
    value = ['use_phone' 'clean' 'converse' 'read_a_book' 'throw_away_trash']
    ix_Class = {k:v for k, v in enumerate(value)}
    model.eval()
    device = checkCuda()
    while True:
        #f,c,h,w
        video = numpy_video[frame_counter]
        
        
        if frame_counter%30==0 and frame_counter!=0:
            input = torch.permute(torch_video[:,prev_counter:frame_counter,:,:],(1,0,2,3)).unsqueeze(0).to(device)
            with torch.no_grad():
                a = model(input).to(device)
                print(a)
                print (torch.argmax(a))
            # print (a)
   
            prev_counter = frame_counter
            # print(input)
         


        cv2.imshow("somevide",video)
        if cv2.waitKey(10) ==ord('q'):
            break
        frame_counter+=1
    pass



def real_time(model):
    device = checkCuda()
    vid = cv2.VideoCapture(0)
    frame_counter = 0
    frames  = None

    resize = Resize((224,224))
    classes = {0:'Using Phone',1:'Cleaning',2:'Reading Book'}
    while True:
        ret,frame = vid.read()
        tensor_frame = torch.Tensor(frame)
        if frames is not None:
            if frames.size(0)==40:
                with torch.no_grad():
                    # frames = frames.unsqueeze(0)
                    frames = torch.permute(frames,(0,3,2,1))
                    frames = resize(frames)
                    frames = frames.unsqueeze(0).to(device)
                    pred = model(frames).to(device)
                    print(classes[torch.argmax(pred).item()])
                    torch.cuda.empty_cache()
                    frames = None

        if frames==None:
        
            frames=tensor_frame.unsqueeze(0)
        else:
            frames = torch.cat([frames,tensor_frame.unsqueeze(0)])
        

        cv2.imshow("test",frame)
        if cv2.waitKey(10)==ord('q'):
            break
        
        # print(frames.size())

# if __name__ == "__main__":
a = torch.randn(30,3,224,224)
model = load_model(r"D:\Activity Recognition - 4th quartile\saved-model\1661592376.671705-val_acc-25.000-acc-45.299-loss-1.093.bin")
# new_model = nn.Sequential(model.model.features)
# out = new_model(a)
# print ((out.view(out.size(0),-1)).size())
# video = predict(r"D:\Activity Recognition - 4th quartile\inference_videos\test_1.mp4",model)
# predict(video,model)
# predict(r'D:\Activity Recognition - 4th quartile\inference_videos\test_2.mp4',model)
# torch.save(model.state_dict(),"saved-model/state-dict.bin")