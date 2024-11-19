import os

def rename_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through each file
    for filename in files:
        # Check if the file contains 'kitchen' in its name
        if 'living' in filename:
            # Construct the new filename by replacing 'kitchen' with 'bathroom'
            new_filename = filename.replace('living', 'desk')

            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


# Example usage:
rename_files('data')