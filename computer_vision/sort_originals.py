import os
images_path = 'original_images/'
annotation_path = 'original_annotations/'
files = os.listdir(images_path)

for index, file in enumerate(files):
   
    base_name = os.path.splitext(file)[0]
    #print(base_name)
    
    exists_path = annotation_path + base_name + '.png'
    
   

    if not os.path.exists(exists_path):
        print("Annotation doesn't exist")
        print(exists_path)
        os.remove(images_path + file)