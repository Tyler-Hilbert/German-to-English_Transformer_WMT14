# Script to download the WMT 2014 English-German dataset
#   Source: https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german#

# Download data
import kagglehub
downloadPath = kagglehub.dataset_download("mohamedlotfy50/wmt-2014-english-german")
print("Downloaded to:", downloadPath)


# Move data to current directory
import os
import shutil
dataPath = "./Data" # Path to data in current directory
print("Moving data to:", dataPath)
os.mkdir(dataPath) # Create directory
shutil.move(os.path.dirname(os.path.dirname(downloadPath)), dataPath) # Move data