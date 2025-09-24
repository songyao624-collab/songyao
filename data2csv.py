# Import the required libraries
from modelscope.msdatasets import MsDataset
import os
import pandas as pd

MAX_DATA_NUMBER = 500

# Check whether the directory already exists
if not os.path.exists('fish_2025_caption'):
    
    ds =  MsDataset.load('modelscope/fish_2025_caption', subset_name='fish_2025_caption', split='train')
    print(len(ds))
    # Set the upper limit of processed image quantity
    total = min(MAX_DATA_NUMBER, len(ds))

    # Create a directory to save images
    os.makedirs('fish_2025_caption', exist_ok=True)

    # Initialize the list of stored image paths and descriptions
    image_paths = []
    captions = []

    for i in range(total):
        # Obtain information for each sample
        item = ds[i]
        image_id = item['image_id']
        caption = item['caption']
        image = item['image']

        # Save the image and record the path
        image_path = os.path.abspath(f'fish_2025_caption/{image_id}.jpg')
        image.save(image_path)

        # Add the path and description to the list
        image_paths.append(image_path)
        captions.append(caption)

        # Print the progress once every 50 images processed
        if (i + 1) % 50 == 0:
            print(f'Processing {i+1}/{total} images ({(i+1)/total*100:.1f}%)')

    # Save the image paths and descriptions as a CSV file
    df = pd.DataFrame({
        'image_path': image_paths,
        'caption': captions
    })

    # Save data as a CSV file
    df.to_csv('./fish-2025-dataset.csv', index=False)

    print(f'Data processing completed. A total of {total} images were processed')

else:
    print('The directory "fish_2025_caption" already exists, skipping data processing step')