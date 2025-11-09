import pandas as pd
from openpyxl import load_workbook
import tarfile
import os
import polars as pl
import dask.dataframe as dd
import zipfile

def load_csv(path, engine='pandas', n_rows=None):
    if engine == 'pandas':
        return pd.read_csv(path, nrows=n_rows)
    elif engine == 'dask':
        df = dd.read_csv(path)
        return df
    elif engine == 'polars':
        return pl.read_csv(path, n_rows=n_rows, quote_char='"')
    else:
        raise ValueError("engine must be 'pandas', 'dask' or 'polars'")

def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")

def load_liar_table(file_path):
   return pd.read_table(file_path,
        names = ['id','label','statement','subject','speaker','job','state','party','barely_true_c','false_c','half_true_c','mostly_true_c','pants_on_fire_c','venue'])

def load_xlsx(file_path):
    wb = load_workbook(filename=file_path)
    sheet = wb.active
    data = list(sheet.values)
    columns = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=columns)
    return df

def extract_tar_gz(tar_path, extract_path):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Files extracted to folder: {extract_to}')

def show_files_in_directory(path):
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return
    print(f"Files in directory {path}:")
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))