import os
import pandas as pd

# Set the root directory where your repositories are located
root_directory = "/home/woody/btr0/btr0104h/data/stl10/RAW"
annotations = pd.read_csv('/home/woody/btr0/btr0104h/data/stl10/stl10_binary/class_names.txt', header=None).values.flatten().tolist()
print(annotations)

# Create or open a text file to write the paths and labels
output_file_path = "/home/woody/btr0/btr0104h/data/stl10/annotations/identity_stl10.txt"
with open(output_file_path, "w") as output_file:
    # Iterate through each subdirectory (class)
    for index, class_directory in enumerate(os.listdir(root_directory), start=1):
        class_path = os.path.join(root_directory, class_directory)
        
        index = annotations.index(class_directory) + 1
        # Check if it's a directory
        if os.path.isdir(class_path):
            
            # Iterate through each image file in the directory
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(".png"):
                    # Write the path and label to the output file
                    
                    image_path = os.path.join('/'.join(class_path.split('/')[-1:]),image_file)
                    output_line = f"{image_path} {index} {class_directory}\n"
                    output_file.write(output_line)

print(f"Output file '{output_file_path}' generated successfully.")