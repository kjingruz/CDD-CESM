import pandas as pd

def load_annotations_csv(csv_file):
    return pd.read_csv(csv_file)

def load_annotations_excel(excel_file):
    return pd.read_excel(excel_file, sheet_name=0)

def compare_classifications(csv_annotations, excel_annotations):
    discrepancies = []

    # Ensure 'Image_name' and 'Pathology Classification/ Follow up' columns exist in the excel_annotations dataframe
    assert 'Image_name' in excel_annotations.columns, "The Excel file must contain 'Image_name' column."
    assert 'Pathology Classification/ Follow up' in excel_annotations.columns, "The Excel file must contain 'Pathology Classification/ Follow up' column."

    for _, row in csv_annotations.iterrows():
        image_name = row['file_name']
        csv_classification = row['category_id']

        excel_row = excel_annotations[excel_annotations['Image_name'].str.contains(image_name, case=False, na=False)]
        if not excel_row.empty:
            excel_classification = excel_row.iloc[0]['Pathology Classification/ Follow up']
            if (excel_classification == 'Benign' and csv_classification != 0) or \
               (excel_classification == 'Malignant' and csv_classification != 1) or \
               (excel_classification != 'Benign' and excel_classification != 'Malignant' and csv_classification != 2):
                discrepancies.append(image_name)
    
    return discrepancies

def main():
    csv_file = '../../data/annotations.csv'
    excel_file = '../../data/Radiology-manual-annotations.xlsx'

    csv_annotations = load_annotations_csv(csv_file)
    excel_annotations = load_annotations_excel(excel_file)
    discrepancies = compare_classifications(csv_annotations, excel_annotations)

    if discrepancies:
        print("Discrepancies found in the following images:")
        for img in discrepancies:
            print(img)
    else:
        print("No discrepancies found.")

if __name__ == "__main__":
    main()
