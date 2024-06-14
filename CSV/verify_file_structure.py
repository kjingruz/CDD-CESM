import os
import pandas as pd

def verify_file_structure(csv_file, image_dirs):
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist.")
        return False
    
    # Load CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return False

    # Verify each image directory
    all_files_exist = True
    for image_dir in image_dirs:
        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} does not exist.")
            return False
        
        # Check if each file listed in the CSV exists in the directory
        for idx, row in df.iterrows():
            filename = os.path.join(image_dir, row['file_name'])
            if not os.path.exists(filename):
                print(f"File {filename} does not exist, skipping.")
                all_files_exist = False
    
    return all_files_exist

if __name__ == "__main__":
    csv_file = '../../data/annotations.csv'
    image_dirs = ['../../data/train', '../../data/valid']

    if verify_file_structure(csv_file, image_dirs):
        print("All files and directories are correctly set up.")
    else:
        print("There are missing files or directories.")
