
from fastai.vision import *
import numpy as np
import pandas as pd
import os
from torchvision import transforms
def fastai_test(path_img,path_model,model_type,img_size,labels,tfms):
  # data path
  datapath = path_img
  modelpath = path_model
  # list to hold images path
  img_names = []
  img_paths = []
  img_labels = []
  img_paths_labels = []
  # collect images name, labels (folder-name) into csv format
  for root, dirs, files in os.walk(datapath):
      for f in files:
          f_path = os.path.join(root,f)
          f_label = os.path.split(root)[-1]
          if not f_label in (labels): continue  
        #if f_label == '.comments' or f_label == 'burn'  : continue
        #rec = f'{f_path},{f_label}'
          img_names.append(f)
          img_paths.append(f_path)
          img_labels.append(f_label)
          img_paths_labels.append([os.path.join(f_label,f),f_label,f_path])

  # images 
  print(f' testing on {len(img_paths_labels)} images:')

  # convert to df to save it as csv
  df = pd.DataFrame(img_paths_labels,columns=['img_names','img_labels','img_path'])
  # file to save image path/name and labels
  df_file = os.path.join(datapath,'test.csv')
  # save to csv 
  df.to_csv(df_file, index=False)
  # print labels
  print(df['img_labels'].value_counts())
  dataset = ImageDataBunch.from_csv(path=datapath  # path having csv_labels file created above
                                ,csv_labels=df_file  # images labeled file name
                                ,ds_tfms=tfms  # transform
                                ,fn_col=0  # indx/name of col having file names 
                                ,label_col=1  # indx/name of col having labels 
                                ,header=1
                                ,size=img_size #256 #128
                                ,bs=20
                                ,valid_pct = 0
                                ).normalize(imagenet_stats) # normalize images

  # show labels
  print(dataset.classes)

  # define the model
  if model_type=='resnet50': model = models.resnet50 
    elif model_type=='resnet34': model = models.resnet34
  learn = cnn_learner(dataset, model, metrics=accuracy)
  learn.data.valid_dl = dataset.train_dl
  learn.data.test_dl = dataset.train_dl

  # load model
  model_file = modelpath
  #model_file = os.path.join(modelpath, 'resnet50_veggie_test')
  learn = learn.load(model_file)
  #End of learner

    #Validate Model : for Classification Interpreter
  learn.validate()

  #Testing Accuracy
  preds, y = learn.get_preds(ds_type=learn.data.test_dl) 
  acc = accuracy(preds,y)
  print(f'Testing Accuracy: {acc.numpy()}')
  return acc.numpy()

