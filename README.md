# Quick_FastAI_Test
A script that helps to quickly test using fastAI.  
-Usage:
```python
fastai_test(img_dir,model_dir,learner_type,img_size,labels,tfms) 
```
-Example:
```python
Testing_Accuracy=fastai_test(img_dir,model_dir,'resnet50',512,['light','medium','dark'],None) 
```
## Step 1
```python
!git clone https://github.com/samnaji/Quick_FastAI_Test.git
sys.path.append('/content/Quick_FastAI_Test')
from fastai_test import *
```

## Step 2
```python
model = models.resnet50
learn = cnn_learner(dataset, model, metrics=accuracy)
learn.fit_one_cycle(25)
learn.save(model_dir, return_path=True)
```
## Step 3
```python
acc=fastai_test(img_dir,model_dir,'resnet50',512,labels,tfms)
```
