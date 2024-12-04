import pandas as pd
import glob
import os

DIR_PATH = '/oscar/data/sbach/bats/projects/ngafid/loci_dataset_fixed_keys/flights'

def flight_paths():

    csv_files = glob.glob(os.path.join(DIR_PATH, "*.csv"))

    data = {}

    for file in csv_files:
        base_name = os.path.basename(file)
        try:
            index_value = int(base_name.split("_")[-1].split(".")[0])
        except ValueError:
            # Skip files that don't match the expected pattern
            print(f"Skipping file {file} as it does not end with '_<number>.csv'")
            continue

        data[index_value] = file

    combined_df = pd.DataFrame.from_dict(data, orient='index', columns=['file_path'])

    combined_df.index.name = 'flight_id'

    return combined_df

