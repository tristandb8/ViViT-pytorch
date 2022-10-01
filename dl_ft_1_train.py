#dataloader for EK CVPR 2022
#dataloader for train
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import config as cfg
import random
import pickle
#import parameters as params

import json
import math
import cv2
# from tqdm import tqdm
import time
import torchvision.transforms as trans
# from decord import VideoReader
from fnmatch import fnmatch
from pathlib import Path
from itertools import chain
import random

class ek_train(Dataset):

    def __init__(self, shuffle = True, trainKitchen = 'p01'):
      print(f'into initiliazation function of DL')
      self.shuffle = shuffle # I still need to add the shuffle functionality
      self.all_paths = self.get_path(trainKitchen)
      if self.shuffle:
            random.shuffle(self.all_paths)
      self.data = self.all_paths
      self.PIL = trans.ToPILImage()
      self.TENSOR = trans.ToTensor()
      self.num_frames = 10 # 10 voxels/clip
      

    def __len__(self):
        return len(self.data)        

    def __getitem__(self,index):
      #I need one clip at a time i.e. 10 voxels
        clip, clip_class,vid_path = self.process_data(index)
        return clip, clip_class,vid_path

    def get_path(self, trainKitchen):
      PATH = []
      folders = [trainKitchen]#, 'p08', 'p22']
      for fol in folders:
        root = '/home/ad358172/AY/event_summer/phase_1/N-EPIC-Kitchens/ek_train_test/train/' + fol + '_train/'
        #pattern = "*.npy"
        for path, subdirs, files in os.walk(root):
            for name in files:
                #if fnmatch(name, pattern):
                PATH.append(path)
        PATH = list(set(PATH))
      PATH.sort()
      #print("Size of Original List",len(PATH))
      #samples = int(len(PATH)*data_pecentage)
      #print(samples)
      #PATH = random.sample(PATH, samples)
      #print("Size of SAMPLED List",len(PATH))

      return PATH
    
    def process_data(self, idx):

        vid_path = self.data[idx].split(' ')[0]
        clip, clip_class = self.build_clip(vid_path)
        return clip, clip_class,vid_path

    def build_clip(self, vid_path):
       clip_class = []
       clip = []
       actions = ['put','take','open','close','wash','cut','mix','pour']
       for id, k in enumerate(actions):
           if(vid_path.find(k)!=-1):
               clip_class = id
               break
       os.chdir(vid_path) #now we are into the parent directory e.g. P01_01 containg all npy voxels
       p = Path.cwd()
       
           	################################ frame list maker starts here ###########################
       files = list(p.glob("*.npy*"))
       files.sort() #sorting in ascending order 
       files = np.array(files)
       frame_count = len(files)
       if(frame_count==self.num_frames):
         s_1 = np.linspace(0,frame_count-1,self.num_frames,dtype=int) # consective 10 voxels
       
       elif(frame_count>self.num_frames):
         start = np.random.randint(frame_count-self.num_frames)
         s_1 = np.linspace(start,start+self.num_frames-1,self.num_frames,dtype=int) 
       elif(frame_count<self.num_frames):
         temp = np.linspace(0,frame_count-1,frame_count,dtype=int)
         s_1 = np.resize(temp, self.num_frames)
       
       #print(frame_count,s_1)
       files_1 = files[s_1]
       for ind, i in enumerate(files_1):
         frame = np.load(i)#frame is the individual voxel
         x = np.einsum('ijk->jki',frame)
         #no clipping done so far
         #x = np.clip(x, a_min = -0.5, a_max = 0.5)
         x = x + np.abs(np.min(x))
         x *= 255/(x.max()) 
         x[x>255] = 255; x[x<0] = 0
         x = x.astype(np.uint8)
         clip.append(self.augmentation(x,(224,224)))
         
       return clip, clip_class  

    def augmentation(self, image, resize_size):
      #image = image.astype(np.float32)
      #image = trans.functional.to_tensor(image)
      #image = image.astype(np.float32)
      image = self.PIL(image)
      transform = trans.transforms.Resize(resize_size)
      image = transform(image)
      image = trans.functional.to_tensor(image) #range 0-1
      #image = torch.tensor(image)
      #image = trans.functional.to_tensor(image)
      #image = image.type(torch.uint8())
      return image
    
def collate_fn2(batch):
  clip = []
  clip_class = []
  vid_path = []
  for item in batch:
        if not (None in item):
            clip.append(torch.stack(item[0],dim=0)) 
            clip_class.append(torch.as_tensor(np.asarray(item[1])))
            vid_path.append(item[2])
      
  clip = torch.stack(clip, dim=0)
  return clip, clip_class,vid_path
    
def vis_frames(clip,name,path):
  #temp = clip[0,:]
  temp = clip.permute(2,3,1,0)
 
  frame_width = 224
  frame_height = 224
  frame_size = (frame_width,frame_height)
  path = path + '/' +  name + '.avi'
  video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('p', 'n', 'g', ' '),2,(frame_size[1],frame_size[0]))
  
  for i in range(temp.shape[3]):
    x = np.array(temp[:,:,:,i])
    x *= 255/(x.max()) 
    x[x>255] = 255
    x[x<0] = 0
    x = x.astype(np.uint8)
    #x = np.clip(x, a_min = -0.5, a_max = 0.5)
    video.write(x) 
  video.release()  
        
def find_action(vid_path):
  actions = ['put','take','open','close','wash','cut','mix','pour']
  for id, k in enumerate(actions):
      if(vid_path.find(k)!=-1):
          clip_class = id
          break
  return clip_class
    
if __name__ == '__main__':
  actions = ['put','take','open','close','wash','cut','mix','pour']
  train_dataset = ek_train(shuffle = True)
  print(f'Train dataset length: {len(train_dataset)}')
  train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle= True,  collate_fn=collate_fn2, drop_last = True)
  t=time.time()
  for i, (clip, clip_class,vid_path) in enumerate(train_dataloader):
    a1 = find_action(vid_path[0])
#   path_output = '/home/adeel/Documents/testing' + actions[a1]
    #path_output = '/home/adeel/Documents/testing' 

    #try:
    #  os.mkdir(path_output)
    #except:
    #  pass
    #vis_frames(clip[0,:],'test',path_output)
   
    #a2 = find_action(vid_path[1])
    #path_output = '/home/adeel/Documents/testing/' + actions[a2]
    #try:
     # os.mkdir(path_output)
    #except:
    #  pass
    #vis_frames(clip[1,:],vid_path[1][-7:],path_output)

    print(i)
  print(f'Time taken to load data is {time.time()-t}')
