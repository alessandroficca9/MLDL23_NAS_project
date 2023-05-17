import wget
import os 
import zipfile 
import shutil
import requests

url = "http://images.cocodataset.org/zips"

train = "train2017.zip"
val = "val2017.zip"

train_download = os.path.join(url, train)
val_download = os.path.join(url,val)

# Create fodler "COCOdataset"
os.mkdir("COCOdataset")


# download train2017.zip
print("downloading train2017.zip ...")
wget.download(url="http://images.cocodataset.org/zips/train2017.zip")


print("Unzipping train2017.zip ...")
with zipfile.ZipFile('train2017.zip',"r") as zip:
    zip.extractall()


shutil.move("train2017", "COCOdataset/")
os.remove("train2017.zip")

# download val2017.zip
print("downloading val2017.zip ...")
wget.download(val_download)

# unzip
print("Unzipping val2017.zip ..")
with zipfile.ZipFile('val2017.zip', 'r') as zip:
    zip.extractall()


shutil.move("val2017","COCOdataset/")
os.remove("val2017.zip")


source_train = 'COCOdataset/train2017'
source_val = "COCOdataset/val2017"
destination = 'COCOdataset/all2017'

os.mkdir(destination)

### Movng all files into "all2017"
print("Moving all files into one folder ...")

# gather all files
allfiles = os.listdir(source_train)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(source_train, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)

print("moved train files")

# gather all files
allfiles = os.listdir(source_val)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(source_val, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)


print("moved val files")

## Download nnotations files
print("Downloading annotations files ...")

annotations_url = "http://images.cocodataset.org/annotations"
ann_instance = "annotations_trainval2017.zip"

wget.download( os.path.join(annotations_url, ann_instance))

with zipfile.ZipFile(ann_instance,"r") as zip:
    zip.extractall()


shutil.move("annotations_trainval2017/annotations", "COCOdataset/")
os.remove(ann_instance)
os.remove("annotations_trainval2017")

