import os
import pandas as pd

def count_images_in_subgroups(base_dir):
    counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    for classification in counts.keys():
        class_dir = os.path.join(base_dir, classification)
        if os.path.exists(class_dir):
            counts[classification] = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        else:
            print(f"Directory {class_dir} does not exist.")
    return counts

def display_counts(train_counts, valid_counts, test_counts):
    total_counts = {key: train_counts[key] + valid_counts[key] + test_counts[key] for key in train_counts.keys()}

    print(f"[{pd.Timestamp.now()} d2.data.build]: Distribution of instances among all 3 categories:")
    print("|  Subset  |  category  | #instances   |")
    print("|:--------:|:----------:|:-------------|")
    print(f"|   Train  |   Benign   | {train_counts['Benign']}          |")
    print(f"|          | Malignant  | {train_counts['Malignant']}         |")
    print(f"|          |   Normal   | {train_counts['Normal']}         |")
    print(f"|          |   total    | {sum(train_counts.values())}         |")
    print(f"|   Valid  |   Benign   | {valid_counts['Benign']}          |")
    print(f"|          | Malignant  | {valid_counts['Malignant']}         |")
    print(f"|          |   Normal   | {valid_counts['Normal']}         |")
    print(f"|          |   total    | {sum(valid_counts.values())}         |")
    print(f"|   Test   |   Benign   | {test_counts['Benign']}          |")
    print(f"|          | Malignant  | {test_counts['Malignant']}         |")
    print(f"|          |   Normal   | {test_counts['Normal']}         |")
    print(f"|          |   total    | {sum(test_counts.values())}         |")
    print(f"|  Overall |   Benign   | {total_counts['Benign']}          |")
    print(f"|          | Malignant  | {total_counts['Malignant']}         |")
    print(f"|          |   Normal   | {total_counts['Normal']}         |")
    print(f"|          |   total    | {sum(total_counts.values())}         |")

if __name__ == "__main__":
    train_dir = '../../data/train'
    valid_dir = '../../data/valid'
    test_dir = '../../data/test'

    # Count images in train, valid, and test subgroups
    train_counts = count_images_in_subgroups(train_dir)
    valid_counts = count_images_in_subgroups(valid_dir)
    test_counts = count_images_in_subgroups(test_dir)

    # Display the counts
    display_counts(train_counts, valid_counts, test_counts)
