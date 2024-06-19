import os
import json

annotations_test_json = os.path.abspath("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test_annotations.json")
test_image_dir = os.path.abspath("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test")

# Load the JSON file
with open(annotations_test_json) as f:
    data = json.load(f)

# Check each image in the JSON file
missing_files = []
for image_info in data['images']:
    category_id = image_info.get('category_id', None)
    if category_id is None:
        print(f"Category ID missing for image_id {image_info['id']}, skipping.")
        continue

    category_name = {1: "benign", 2: "malignant", 3: "normal"}[category_id]
    file_path = os.path.join(test_image_dir, category_name, image_info['file_name'])
    if not os.path.exists(file_path):
        missing_files.append(file_path)
    else:
        image_info['file_name'] = os.path.join(category_name, image_info['file_name'])

# Print the missing files
if missing_files:
    print("The following files are missing:")
    for file in missing_files:
        print(file)
else:
    print("All files are present.")

# Optional: Save the corrected JSON
corrected_json_path = os.path.abspath("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test_annotations_corrected.json")
with open(corrected_json_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Corrected JSON file saved to {corrected_json_path}")
