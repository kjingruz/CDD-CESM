import os
import pandas as pd

def count_images_in_csv(csv_file):
    df = pd.read_csv(csv_file)
    counts = df['category_id'].value_counts().to_dict()
    # Initialize counts for all categories
    counts = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
    # Map numerical categories to their names
    category_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
    named_counts = {category_mapping[key]: value for key, value in counts.items()}
    return named_counts

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
    train_csv = '../../data/train_annotations.csv'
    valid_csv = '../../data/valid_annotations.csv'
    test_csv = '../../data/test_annotations.csv'

    # Count images in train, valid, and test CSV files
    train_counts = count_images_in_csv(train_csv)
    valid_counts = count_images_in_csv(valid_csv)
    test_counts = count_images_in_csv(test_csv)

    # Display the counts
    display_counts(train_counts, valid_counts, test_counts)
