import pandas as pd
from collections import Counter
import random
import os
import shutil
from tqdm import tqdm


# Step 4: Identify images belonging to exactly one disease
# Assuming each disease is a separate label in the 'Finding Labels' column
def single_disease_images(data):
    single_disease_images = []
    for index, row in data.iterrows():
        diseases = row["Finding Labels"].split('|')
        if len(diseases) == 1:
            single_disease_images.append((row["Image Index"], diseases[0]))
    return single_disease_images

# Define function to create folder structure
def create_folder_structure(samples, source_folder, target_folder):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Iterate through samples and copy images to corresponding subfolders
    for filename, disease in tqdm(samples):
        # Create disease subfolder if it doesn't exist
        disease_folder = os.path.join(target_folder, disease)
        if not os.path.exists(disease_folder):
            os.makedirs(disease_folder)
        
        # Copy image file to disease subfolder
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(disease_folder, filename)
        shutil.copyfile(source_path, target_path)


if __name__ == "__main__":
    # Step 1: Read the CSV file
    data = pd.read_csv("../.data/dataset/Data_Entry_2017_v2020.csv")

    # Step 2: Read train and test image lists
    with open("../.data/dataset/train_val_list.txt", "r") as train_file:
        train_list = [line.strip() for line in train_file]

    with open("../.data/dataset/test_list.txt", "r") as test_file:
        test_list = [line.strip() for line in test_file]

    # Step 3: Filter dataset to include only train and test images
    train_data = data[data["Image Index"].isin(train_list)]
    test_data = data[data["Image Index"].isin(test_list)]

    # Step 5: Create subset with original distribution
    train_subset = single_disease_images(train_data)
    test_subset = single_disease_images(test_data)

    # Print subsets (double check)
    print("Train Subset:")
    print(train_subset[:10])  # Printing first 10 examples
    print("Test Subset:")
    print(test_subset[:10])  # Printing first 10 examples
    # Example output:
    # Train Subset:
    # [('00000001_000.png', 'Cardiomegaly'), ('00000002_000.png', 'No Finding'), ('00000005_000.png', 'No Finding'), ('00000005_001.png', 'No Finding'), ('00000005_002.png', 'No Finding'), ('00000005_003.png', 'No Finding'), ('00000005_004.png', 'No Finding'), ('00000005_005.png', 'No Finding'), ('00000005_006.png', 'Infiltration'), ('00000006_000.png', 'No Finding')]
    # Test Subset:
    # [('00000003_001.png', 'Hernia'), ('00000003_002.png', 'Hernia'), ('00000003_004.png', 'Hernia'), ('00000003_005.png', 'Hernia'), ('00000003_006.png', 'Hernia'), ('00000003_007.png', 'Hernia'), ('00000003_000.png', 'Hernia'), ('00000013_024.png', 'Mass'), ('00000013_033.png', 'Pneumothorax'), ('00000013_036.png', 'Pneumothorax')]


    # Count frequencies of each disease in train_subset
    train_disease_counts = Counter([disease for _, disease in train_subset])

    # Count frequencies of each disease in test_subset
    test_disease_counts = Counter([disease for _, disease in test_subset])

    train_disease_ratio = {}
    test_disease_ratio = {}

    # Print distributions (Double check)
    print("Train Subset Distribution:")
    for disease, count in train_disease_counts.items():
        print(f"{disease}: {count}")
        train_disease_ratio[disease] = count / len(train_subset)

    print("\nTest Subset Distribution:")
    for disease, count in test_disease_counts.items():
        print(f"{disease}: {count}")
        test_disease_ratio[disease] = count / len(test_subset)
    # Example output:
    # Train Subset Distribution:
    # Cardiomegaly: 777
    # No Finding: 50500
    # Infiltration: 7327
    # Nodule: 2248
    # Emphysema: 587
    # Effusion: 2788
    # Atelectasis: 3414
    # Pleural_Thickening: 817
    # Fibrosis: 551
    # Mass: 1696
    # Pneumonia: 234
    # Pneumothorax: 1241
    # Hernia: 65
    # Consolidation: 829
    # Edema: 397

    # Test Subset Distribution:
    # Hernia: 45
    # Mass: 443
    # Pneumothorax: 953
    # No Finding: 9861
    # Emphysema: 305
    # Cardiomegaly: 316
    # Infiltration: 2220
    # Pleural_Thickening: 309
    # Effusion: 1167
    # Consolidation: 481
    # Edema: 231
    # Atelectasis: 801
    # Fibrosis: 176
    # Pneumonia: 88
    # Nodule: 457


    # Number of samples to select
    train_samples = 10000
    test_samples = 2000

    # Select training samples
    selected_train_samples = random.choices(
        population=train_subset,
        weights=[train_disease_ratio[disease] for _, disease in train_subset],
        k=train_samples
    )

    # Select test samples
    selected_test_samples = random.choices(
        population=test_subset,
        weights=[test_disease_ratio[disease] for _, disease in test_subset],
        k=test_samples
    )

    # Print selected samples (double check)
    print("Selected Training Samples:")
    print(selected_train_samples[:10])  # Print first 10 selected training samples
    print("\nSelected Test Samples:")
    print(selected_test_samples[:10])   # Print first 10 selected test samples
    # Example output:
    # Selected Training Samples:
    # [('00000086_000.png', 'No Finding'), ('00022142_000.png', 'No Finding'), ('00016131_000.png', 'No Finding'), ('00013054_000.png', 'No Finding'), ('00027648_015.png', 'No Finding'), ('00009172_000.png', 'No Finding'), ('00014994_000.png', 'No Finding'), ('00017546_001.png', 'No Finding'), ('00007274_004.png', 'No Finding'), ('00006700_000.png', 'No Finding')]

    # Selected Test Samples:
    # [('00027464_032.png', 'No Finding'), ('00012045_026.png', 'No Finding'), ('00012834_036.png', 'Effusion'), ('00010012_003.png', 'No Finding'), ('00030050_000.png', 'No Finding'), ('00006948_010.png', 'No Finding'), ('00028257_000.png', 'No Finding'), ('00028344_000.png', 'No Finding'), ('00003973_010.png', 'Infiltration'), ('00030375_001.png', 'No Finding')]

    # Count occurrences of each disease in selected_train_samples
    a = Counter([disease for _, disease in selected_train_samples])

    # Count occurrences of each disease in selected_test_samples
    b = Counter([disease for _, disease in selected_test_samples])

    # Print distributions
    print("Train Selected Subset Distribution:")
    for disease, count in a.items():
        print(f"{disease}: {count}")

    print("\nTest Selected Subset Distribution:")
    for disease, count in b.items():
        print(f"{disease}: {count}")
    # Example output:
    # Train Selected Subset Distribution:
    # No Finding: 9655
    # Infiltration: 225
    # Pneumothorax: 9
    # Atelectasis: 37
    # Effusion: 34
    # Nodule: 23
    # Mass: 8
    # Fibrosis: 1
    # Cardiomegaly: 4
    # Pleural_Thickening: 3
    # Consolidation: 1

    # Test Selected Subset Distribution:
    # No Finding: 1849
    # Effusion: 29
    # Infiltration: 84
    # Atelectasis: 14
    # Nodule: 5
    # Emphysema: 1
    # Pneumothorax: 10
    # Mass: 5
    # Consolidation: 3
    

    # Define source and target folders
    source_folder = "../.data/dataset/images"
    train_target_folder = "../.data/dataset/train"
    test_target_folder = "../.data/dataset/test"

    # Create train folder structure
    create_folder_structure(selected_train_samples, source_folder, train_target_folder)

    # Create test folder structure
    create_folder_structure(selected_test_samples, source_folder, test_target_folder)

    print("Folder structure created successfully.")
