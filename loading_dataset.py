# Downloading dataset:
from zipfile import ZipFile
import os
os.environ['KAGGLE_USERNAME'] = "KAGGLE_USERNAME" # username from the json file
os.environ['KAGGLE_KEY'] = "KAGGLE_KEY" # key from the json file
!kaggle datasets download -d praveengovi/coronahack-chest-xraydataset # api copied from kaggle

# Create a ZipFile Object and load chest-xray-pneumonia.zip in it
with ZipFile('chest-xray-pneumonia.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
   
# KAGGLE LINK:- https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

import os
import pandas as pd
import time
import shutil 

PATH_TRAIN = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
TOTAL_IMGS = len(os.listdir(PATH_TRAIN))
normal = 0
infected = 0
data = pd.read_csv('Chest_xray_Corona_Metadata.csv')


img = data['X_ray_image_name']
label = data['Label']
image_type = data['Dataset_type']
all_dir = os.listdir(PATH_TRAIN)

os.mkdir(PATH_TRAIN+"/INFECTED")
os.mkdir(PATH_TRAIN+"/NORMAL")

wrong_info = 0 # Checking if the provided list maps the images correctly

# Moving the train images to designated folders

for i in range(len(image_type)):
    if image_type[i] == "TRAIN":
        if img[i] in all_dir:
            if label[i]=="Normal":
                infected = infected+1
                shutil.move(PATH_TRAIN+'/'+img[i], PATH_TRAIN+'/'+'NORMAL/'+img[i])
                normal = normal + 1

            else:
                
                shutil.move(PATH_TRAIN+'/'+img[i], PATH_TRAIN+'/'+'INFECTED/'+img[i])
                infected = infected +1

        else:
            wrong_info = wrong_info +1  

print("X-ray of Normal patients (TRAIN DATASET): "+str(normal),"X-ray of Infected patients (TRAIN DATASET): "+str(infected))

# Visualization of train dataset

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
categories = ['NORMAL','INFECTED']
number_of_imgs = [normal,infected]
ax.bar(0,number_of_imgs[0],color = 'g',width = 0.1)
ax.bar(0.15,number_of_imgs[1],color = 'r',width = 0.1)
ax.legend(labels=categories)
ax.set_ylabel('Number of images')
ax.set_xlabel('Categories')
plt.show()


PATH_TEST = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
TOTAL_IMGS = len(os.listdir(PATH_TEST))
normal = 0
infected = 0

img = data['X_ray_image_name']
label = data['Label']
image_type = data['Dataset_type']
all_dir = os.listdir(PATH_TEST)

os.mkdir(PATH_TEST+"/INFECTED")
os.mkdir(PATH_TEST+"/NORMAL")

wrong_info = 0

# Moving the test images to designated folders

for i in range(len(image_type)):
    if image_type[i] == "TEST":
        if img[i] in all_dir:
            if label[i]=="Normal":
                infected = infected+1
                shutil.move(PATH_TEST+'/'+img[i], PATH_TEST+'/'+'NORMAL/'+img[i])
                normal = normal + 1

            else:
                
                shutil.move(PATH_TEST+'/'+img[i], PATH_TEST+'/'+'INFECTED/'+img[i])
                infected = infected +1

        else:
            wrong_info = wrong_info +1  

print("X-ray of Normal patients (TEST DATASET): "+str(normal),"X-ray of Infected patients (TEST DATASET): "+str(infected))

# Visualization of test dataset

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
categories = ['NORMAL','INFECTED']
number_of_imgs = [normal,infected]
ax.bar(0,number_of_imgs[0],color = 'g',width = 0.1)
ax.bar(0.15,number_of_imgs[1],color = 'r',width = 0.1)
ax.legend(labels=categories)
ax.set_ylabel('Number of images')
ax.set_xlabel('Categories')
plt.show()
        
        
