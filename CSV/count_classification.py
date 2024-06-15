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

def display_counts(train_counts, test_counts):
    total_counts = {key: train_counts[key] + test_counts[key] for key in train_counts.keys()}

    print(f"[{pd.Timestamp.now()} d2.data.build]: Distribution of instances among all 3 categories:")
    print("|  category  | #instances   |  category  | #instances   |  category  | #instances   |")
    print("|:----------:|:-------------|:----------:|:-------------|:----------:|:-------------|")
    print(f"|   Benign   | {train_counts['Benign']}          | Malignant  | {train_counts['Malignant']}         |   Normal   | {train_counts['Normal']}         |")
    print(f"|   total    | {sum(train_counts.values())}         |            |              |            |              |")
    print(f"|   Benign   | {test_counts['Benign']}          | Malignant  | {test_counts['Malignant']}         |   Normal   | {test_counts['Normal']}         |")
    print(f"|   total    | {sum(test_counts.values())}         |            |              |            |              |")
    print(f"|   Benign   | {total_counts['Benign']}          | Malignant  | {total_counts['Malignant']}         |   Normal   | {total_counts['Normal']}         |")
    print(f"|   total    | {sum(total_counts.values())}         |            |              |            |              |")

if __name__ == "__main__":
    train_dir = '../../data/train'
    test_dir = '../../data/test'

    # Count images in train and test subgroups
    train_counts = count_images_in_subgroups(train_dir)
    test_counts = count_images_in_subgroups(test_dir)

    # Display the counts
    display_counts(train_counts, test_counts)
