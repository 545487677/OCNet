import subprocess
import os 
import numpy as np
from multiprocessing import Pool
import re
from rdkit import Chem
import multiprocessing
from tqdm import tqdm
import lmdb
import pickle
import pandas as pd 
from rdkit.Chem import AllChem

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_NAME_LL = "film_ll"

def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def process_data(args):
    i, _idx, coord_list, atom_list, name, features = args
    _coord = np.array(coord_list)
    _atom = [normalize_atoms(a) for a in np.array(atom_list)]
    assert len(_coord) == len(_atom), 'length of coord and atom should be the same'
    if features is None:
        return None
    # Check if any element within the four-dimensional features array contains None
    def check_nested_none(arr):
        if isinstance(arr, list):
            return any(check_nested_none(x) for x in arr)
        return arr is None

    if np.isnan(features).any():
        # print(f"NaN detected in features for ID {i}. Replacing NaN with 0.")
        features = np.nan_to_num(features, nan=0.0)

    if check_nested_none(features):
        return None

    dd = {
        'ID': i,
        'coordinates': [_coord],
        'atoms': _atom,
        'atoms_charge': None,
        'target' : (0.0,),
        'num_atoms': len(_atom),
        'features': features,
        'mol_name': (name, ),
        'idx': name.split('_')[-1],
    }

    dd_pkl = pickle.dumps(dd, protocol=-1)
    return f'{i}'.encode("ascii"), dd_pkl

def process_to_lmdb(input_path, output_path, task_name):
    os.makedirs(output_path, exist_ok=True)
    data = np.load(input_path, allow_pickle=True)
    coord = data['coord']
    symbol = data['symbol']
    if task_name == 'hh':
        features = data['feature_hh'].astype(float)
    elif task_name == 'll':
        features = data['feature'].astype(float)
    else:
        print('task name not recognized')
    mol_name = data['name']

    print("shape: \n", )
    print(coord.shape)
    print(symbol.shape)
    np.random.seed(42)
    idx = np.random.permutation(range(len(symbol)))
    _, val_ratio = 0, 1.0
    val_idx = idx[:int(len(symbol)*val_ratio)]
    train_idx = idx[int(len(symbol)*val_ratio):]

    nthreads = multiprocessing.cpu_count()
    print("Number of CPU cores:", nthreads)

    for name, idx in [('valid.lmdb', val_idx)]:
        outputfilename = os.path.join(output_path, name)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(1024*1024*1024*50),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            args_list = [(i, _idx, coord[_idx], symbol[_idx], mol_name[_idx],  features[_idx]) 
             for i, _idx in enumerate(tqdm(idx))]
             
            for result in pool.imap(process_data, args_list):
                if result is not None:
                    key, inner_output = result
                    txn_write.put(key, inner_output)
                    i += 1
                    if i % 1000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()   
    return

def workflow(input_npz_path=None, output_path=None):
    os.makedirs(output_path, exist_ok=True)
    # ll
    output_path_ll = os.path.join(output_path, TASKS_NAME_LL)
    process_to_lmdb(input_path=input_npz_path, output_path=output_path_ll, task_name='ll')
    lmdb_file = os.path.join(output_path_ll, "valid.lmdb")
    if not os.path.exists(lmdb_file):
        raise FileNotFoundError(f"LMDB file not found: {lmdb_file}")

if __name__ == "__main__":

    input_path = '/vepfs/fs_users/zhengcheng/unimol_learn/OCNet/biomolecular_properties/data/film_elec_mobility/mol_105901_mob/data_ll.npz'
    output_path = '/vepfs/fs_projects/FunMG/paper_code/OCNet/biomolecular_properties/data/film_ll_elec/'
    workflow(input_path, output_path)