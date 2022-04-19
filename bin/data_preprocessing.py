import re
import os
import gdown
import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
from pyzipper import AESZipFile
from padelpy import from_sdf, from_smiles
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import PandasTools, MolToSmiles, MolFromSmiles
from progress.bar import FillingSquaresBar, IncrementalBar
from imblearn.over_sampling import RandomOverSampler

ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = Path(ROOT, 'bin')
OUTPUT_PATH = Path(ROOT, 'output')
DATASET_PATH = Path(ROOT, 'dataset')
IMAGES_PATH = Path(DATASET_PATH, 'images')
DATASET_FILENAME = 'P-gp_act_inact_dataset.csv'
CHANNELS = 1  # 1: GRAY | 3: RGB


def str_list(string):
    """Convert string to list numeric 
    Ex: '[1,2,3,4]' 
    ---->[1,2,3,4]
    """

    pattern = '\d+'
    result = re.findall(pattern, string)
    return [*map(lambda x: int(x), result)]


def preprocess_input(img_path, target_size, channels):
    raw = tf.io.read_file(img_path)
    image = tf.image.decode_png(raw, channels=channels)
    image = tf.image.resize(image, target_size)
    return image


def preprocessing_image_pipeline(image_size):
    def main_preprocessing(image_path, label):
        image = preprocess_input(image_path, image_size, CHANNELS)
        return (image, label)

    return main_preprocessing


def download_dataset(file_id, dest_path):
    gdown.download(id=file_id, output=dest_path)


def unzip_dataset(zip_path, dest_path, pwd=None):
    with AESZipFile(zip_path) as f:
        if pwd is not None:
            f.pwd = bytes(pwd, 'utf8')
        f.extractall(dest_path)


def make_folder(parents_path, folder_name):
    dest_path = Path(parents_path, folder_name)
    if not os.path.isdir(dest_path):
        print(f'Creating {folder_name} folder...')
        os.mkdir(dest_path)


def create_image(image_path, name, smile):
    # print(f'Creating {name}.png')
    img_path = image_path
    mol = MolFromSmiles(smile)
    img = Draw.MolToImage(mol)
    img.save(img_path)


def preprocessing():
    """ 
    Create dataset csv format
    """
    # create dataset folder
    make_folder(ROOT, 'dataset')
    # create images dataset folder
    make_folder(DATASET_PATH, 'images')
    # create output folder
    make_folder(ROOT, 'output')
    # create images output folder
    make_folder(OUTPUT_PATH, 'images')

    # read chembl4302.xlsx
    df_chembl4302 = pd.read_excel(
        Path(DATASET_PATH, 'ChemBL4302.xlsx'), sheet_name='ChemBL4302')
    df_chembl4302.drop(
        columns=['Molecule Name', 'Standard Value', 'Standard Units'], inplace=True)
    # df_chembl4302.drop_duplicates()

    # read ecker_Pgp.xlsx
    df_ecker = pd.read_excel(
        Path(DATASET_PATH, 'Ecker_Pgp.xlsx'), sheet_name='Ecker_Pgp')
    df_ecker.drop(
        columns=['Real Name'], inplace=True)

    df = pd.concat([df_chembl4302, df_ecker])  # concat chembl4302 vs ecker
    df['Path'] = df['Name'] + '.png'
    df['Path'] = df['Path'].map(lambda x: Path(IMAGES_PATH, x))
    # Create images
    bar = IncrementalBar('Creating images', max=len(df))
    for i, row in df.iterrows():
        create_image(row['Path'], row['Name'], row['Smile'])
        bar.next()
    bar.finish()

    # EDA
    print('Raw dataset')
    raw_act = df[df['Activity'] == 1]
    raw_inact = df[df['Activity'] == 0]
    print('Number of Activity:', raw_act.shape[0])
    print('Number of Inactivity:', raw_inact.shape[0])
    print('Random Oversampling...')

    # Random Oversampling
    ros = RandomOverSampler()
    X = df.drop(['Activity'], axis=1)
    y = df['Activity']
    X_ros, y_ros = ros.fit_resample(X, y)
    ros_df = pd.concat([X_ros, y_ros], axis=1)
    ros_act = ros_df[ros_df['Activity'] == 1]
    ros_inact = ros_df[ros_df['Activity'] == 0]
    print('Number of ROS Activity:', ros_act.shape[0])
    print('Number of ROS Inactivity:', ros_inact.shape[0])

    # save dataset
    df.to_csv(Path(DATASET_PATH, 'raw_'+DATASET_FILENAME), index=False)
    ros_df.to_csv(Path(DATASET_PATH, DATASET_FILENAME), index=False)


if __name__ == '__main__':
    preprocessing()
