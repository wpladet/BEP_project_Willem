import os
import h5py
import torch

def split_h5_file(input_filepath, output_dir1, output_dir2):
    """
    Splits the features from an HDF5 file and saves them as a new .h5 file.
    Saves the coords as a PyTorch tensor file with .pt extension.

    Args:
    input_filepath (str): Path to the input HDF5 file.
    output_dir (str): Directory to save the split files.

    Returns:
    None
    """
    # Extract the base name of the file without extension
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    
    # Define output file paths
    features_filepath = os.path.join(output_dir1, f'{base_filename}.h5')
    coords_filepath = os.path.join(output_dir2, f'{base_filename}.pt')
    
    # Open the original HDF5 file
    with h5py.File(input_filepath, 'r') as file:
        # Access the features and coords datasets
        features = file['features'][:]
        coords = file['coords'][:]
    
    # Create a new HDF5 file for features and save
    with h5py.File(features_filepath, 'w') as f:
        f.create_dataset('features', data=features)

    # Convert coords to PyTorch tensor and save as .pt file
    coords_tensor = torch.from_numpy(coords)
    torch.save(coords_tensor, coords_filepath)

    print(f'Split {input_filepath} into {features_filepath} and {coords_filepath}')

def split_all_h5_files_in_folder(folder_path, output_dir1, output_dir2):
    """
    Splits all .h5 files in a given folder into separate features and coords files.
    Features remain as .h5 files, coords are converted to .pt files.

    Args:
    folder_path (str): Path to the folder containing the .h5 files.
    output_dir (str): Directory to save the split files.

    Returns:
    None
    """
    # Make sure output directory exists
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    # Make sure output directory exists
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)

    # List all .h5 files in the directory
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    # Split each file
    for h5_file in h5_files:
        input_filepath = os.path.join(folder_path, h5_file)
        split_h5_file(input_filepath, output_dir1, output_dir2)

folder_path = '/home/mcs001/20181133/CLAM/data_feat/Camelyon16_patch256_ostu_res50/h5_files_original'
output_dir1 = '/home/mcs001/20181133/CLAM/data_feat/Camelyon16_patch256_ostu_res50/h5_files'
output_dir2 = '/home/mcs001/20181133/CLAM/data_feat/Camelyon16_patch256_ostu_res50/coords'
split_all_h5_files_in_folder(folder_path, output_dir1, output_dir2)
