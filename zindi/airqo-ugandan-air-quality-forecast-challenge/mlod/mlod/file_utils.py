import requests
import os
from tqdm import tqdm
from pathlib import Path

import logging
logger = logging.getLogger('mlod')

def download_from_url(download_url: str, store_dir: str = './', save_name_file: str = None, **req_config: dict):
    """Downloads a file from url"""
    x = requests.post(download_url, data = req_config, stream=True)

    if save_name_file is None:
        save_name_file = download_url.split('/')[-1]

    # Get the section of the path
    path = Path(store_dir).joinpath(save_name_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    

    # write to the file
    with path.open("wb") as fl:
        logger.info(f"Downloading and saving to {path}")

        for chunk in tqdm(x.iter_content(chunk_size=512)):
            if chunk:  # filter out keep-alive new chunks
                fl.write(chunk)

    # returns the full path that contains the file
    return path

class PredictionStorage:
    @staticmethod
    def save_csv_from_dict(csv_file_path: str, dict_to_save: dict):
        import pandas as pd

        save_df = pd.DataFrame.from_dict(dict_to_save)
        save_df.to_csv(csv_file_path)

        logger.info('Saved to \'{}\''.format(csv_file_path))
        
        del save_df