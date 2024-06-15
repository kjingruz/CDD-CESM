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

def verify_and_copy_files(excel_file, image_dir, train_dir, test_dir, train_ratio=0.8):
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"Excel file {excel_file} does not exist.")
        return False

    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name=0)  # Ensure correct sheet name
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return False

    # Ensure the dataframe has the required columns
    required_columns = ['Image_name', 'Pathology Classification/ Follow up']
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain 'Image_name' and 'Pathology Classification/ Follow up' columns.")
        return False

    # Rename the classification column for consistency
    df.rename(columns={'Pathology Classification/ Follow up': 'classification', 'Image_name': 'file_name'}, inplace=True)

    # Normalize file extensions
    df['file_name'] = df['file_name'].apply(lambda x: f"{x}.jpg" if not x.lower().endswith('.jpg') else x)

    # Extract patient ID and add it to the dataframe
    df['patient_id'] = df['file_name'].apply(extract_patient_id)

    # Group by patient_id
    grouped = df.groupby('patient_id')

    # Create a list of groups (each group corresponds to one patient)
    patient_groups = [group for _, group in grouped]

    # Split the patient groups into training and validation sets
    train_groups, test_groups = train_test_split(patient_groups, train_size=train_ratio, random_state=42)

    def copy_files(groups, target_dir):
        all_files_exist = True
        missing_files = []
        for group in groups:
            for _, row in group.iterrows():
                filename = row['file_name']
                classification = row['classification']
                src_path = os.path.join(image_dir, filename)
                dst_dir = os.path.join(target_dir, classification)
                dst_path = os.path.join(dst_dir, filename)

                if not os.path.exists(src_path):
                    print(f"File {src_path} does not exist, skipping.")
                    missing_files.append(src_path)
                    all_files_exist = False
                else:
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")

        if missing_files:
            print("Missing files:")
            for file in missing_files:
                print(file)
                
        return all_files_exist, missing_files

    # Verify and copy training files
    print("Copying training files...")
    train_files_exist, missing_train_files = copy_files(train_groups, train_dir)
    # Verify and copy validation files
    print("Copying validation files...")
    test_files_exist, missing_test_files = copy_files(test_groups, test_dir)

    # Attempt to copy missing files again
    if not train_files_exist or not test_files_exist:
        print("Re-attempting to copy missing files...")
        for missing_file in missing_train_files:
            src_path = os.path.join(image_dir, os.path.basename(missing_file))
            classification = df[df['file_name'] == os.path.basename(missing_file)]['classification'].values[0]
            dst_dir = os.path.join(train_dir, classification)
            dst_path = os.path.join(dst_dir, os.path.basename(missing_file))
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
                print(f"Re-attempted copy: {src_path} to {dst_path}")

        for missing_file in missing_test_files:
            src_path = os.path.join(image_dir, os.path.basename(missing_file))
            classification = df[df['file_name'] == os.path.basename(missing_file)]['classification'].values[0]
            dst_dir = os.path.join(test_dir, classification)
            dst_path = os.path.join(dst_dir, os.path.basename(missing_file))
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
                print(f"Re-attempted copy: {src_path} to {dst_path}")

    return train_files_exist and test_files_exist

def verify_classifications(excel_file):
    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name=0)  # Ensure correct sheet name
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return False

    # Ensure the dataframe has the required columns
    required_columns = ['Image_name', 'Pathology Classification/ Follow up']
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain 'Image_name' and 'Pathology Classification/ Follow up' columns.")
        return False

    # Rename the classification column for consistency
    df.rename(columns={'Pathology Classification/ Follow up': 'classification', 'Image_name': 'file_name'}, inplace=True)

    # Verify classifications
    for _, row in df.iterrows():
        filename = row['file_name']
        classification = row['classification']
        if classification not in ['Benign', 'Malignant', 'Normal']:
            print(f"Invalid classification {classification} for file {filename}")
            return False

    print("All classifications are verified.")
    return True

if __name__ == "__main__":
    excel_file = '../../data/Radiology-manual-annotations.xlsx'
    image_dir = '../../data/images'
    train_dir = '../../data/train'
    test_dir = '../../data/test'

    # Create directories if they don't exist
    create_directories([train_dir, test_dir])

    # Verify classifications
    if verify_classifications(excel_file):
        # Verify files and copy them to the correct directories
        if verify_and_copy_files(excel_file, image_dir, train_dir, test_dir):
            print("All files are correctly set up.")
        else:
            print("There are missing files or directories.")
    else:
        print("There are issues with the classifications.")
