import os
path = 'validation/images'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(path + "/" + file, path + "/" + f"tf_{file}")