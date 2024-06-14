import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def extract_patient_id(filename):
    return filename.split('_')[0][1:]

def verify_and_copy_files(excel_file, image_dir, train_dir, valid_dir, train_ratio=0.8):
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"Excel file {excel_file} does not exist.")
        return False

    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return False

    # Ensure the dataframe has the required columns
    required_columns = ['file_name', 'Pathology Classification/ Follow up']
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain 'file_name' and 'Pathology Classification/ Follow up' columns.")
        return False

    # Rename the classification column for consistency
    df.rename(columns={'Pathology Classification/ Follow up': 'classification'}, inplace=True)

    # Extract patient ID and add it to the dataframe
    df['patient_id'] = df['file_name'].apply(extract_patient_id)

    # Group by patient_id
    grouped = df.groupby('patient_id')

    # Create a list of groups (each group corresponds to one patient)
    patient_groups = [group for _, group in grouped]

    # Split the patient groups into training and validation sets
    train_groups, valid_groups = train_test_split(patient_groups, train_size=train_ratio, random_state=42)

    def copy_files(groups, target_dir):
        all_files_exist = True
        for group in groups:
            for _, row in group.iterrows():
                filename = row['file_name']
                classification = row['classification']
                src_path = os.path.join(image_dir, filename)
                dst_dir = os.path.join(target_dir, classification)
                dst_path = os.path.join(dst_dir, filename)

                if not os.path.exists(src_path):
                    print(f"File {src_path} does not exist, skipping.")
                    all_files_exist = False
                else:
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")

        return all_files_exist

    # Verify and copy training files
    train_files_exist = copy_files(train_groups, train_dir)
    # Verify and copy validation files
    valid_files_exist = copy_files(valid_groups, valid_dir)

    return train_files_exist and valid_files_exist

if __name__ == "__main__":
    excel_file = '../../data/Radiology-manual-annotations.xlsx'
    image_dir = '../../data/images'
    train_dir = '../../data/train'
    valid_dir = '../../data/valid'

    # Create directories if they don't exist
    create_directories([train_dir, valid_dir])

    # Verify files and copy them to the correct directories
    if verify_and_copy_files(excel_file, image_dir, train_dir, valid_dir):
        print("All files are correctly set up.")
    else:
        print("There are missing files or directories.")
