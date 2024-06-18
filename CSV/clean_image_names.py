import os

folder_path = '../../data/images/'

for filename in os.listdir(folder_path):
    # Normalize the filename by stripping leading/trailing spaces and reducing multiple spaces to a single space
    filename_new = ' '.join(filename.split())
    if not filename_new.endswith('.jpg'):
        filename_new += '.jpg'
    else:
        filename_new = filename_new[:-4].strip() + '.jpg'
    
    # Generate full source and destination paths
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, filename_new)

    # Rename the file if the new filename is different
    if src != dst:
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")

print("Filenames normalization completed.")
