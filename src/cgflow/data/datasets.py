from abc import ABC, abstractmethod
from pathlib import Path

import os
import shutil

import numpy as np
import pickle
import torch
import lmdb
from tqdm import tqdm
from cgflow.util.molrepr import GeometricMol, GeometricMolBatch
from cgflow.util.pocket import PocketComplex, PocketComplexBatch, ProteinPocket

# *** Util functions ***


def load_smol_data(data_path, smol_cls):
    data_path = Path(data_path)

    combined_bytes = b""
    if data_path.is_dir():
        for file in sorted(data_path.iterdir()):  # Sort for consistent order
            if file.is_file() and file.suffix == ".smol":
                combined_bytes += file.read_bytes()
    else:
        # TODO maybe read in chunks if this is too big
        combined_bytes = data_path.read_bytes()

    return smol_cls.from_bytes(combined_bytes)


# *** Abstract class for all Smol data types ***


class SmolDataset(ABC, torch.utils.data.Dataset):

    def __init__(self, smol_data, transform=None):
        super().__init__()

        self._data = smol_data
        self.transform = transform
        self.lengths = self._data.seq_length

    @property
    def hparams(self):
        return {}

    def __len__(self):
        return self._data.batch_size

    def __getitem__(self, item):
        molecule = self._data[item]
        if self.transform is not None:
            molecule = self.transform(molecule)

        return molecule

    @classmethod
    @abstractmethod
    def load(cls, data_path, transform=None):
        pass


class LMDBSmolDataset(ABC, torch.utils.data.Dataset):

    def __init__(self, key_path, lmdb_path, max_length=None, transform=None):
        super().__init__()
        self.max_length = max_length if max_length is not None else float(
            'inf')

        self.keys = []
        with open(key_path, 'r') as f:
            for line in f:
                self.keys.append(line.strip())
        self.transform = transform

        self.lmdb_path = str(lmdb_path)
        self.env = lmdb.open(self.lmdb_path,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             max_readers=512)

        self.length_path = Path(self.lmdb_path) / 'lengths.pkl'
        if self.length_path.exists():
            self.lengths = pickle.load(open(self.length_path, 'rb'))
            assert len(self.lengths) == len(self.keys)
        else:
            self.lengths = []
            for i in tqdm(range(len(self.keys))):
                with self.env.begin(write=False) as txn:
                    value = txn.get(self.keys[i].encode('utf-8'))
                    self.lengths.append(len(value) * 0.009)
                # Instead of acutally loading the object which takes a long time -
                # We'll just directly estimate the length of object based on the number
                # of bytes
                # item = self.__getitem__(i)
                # self.lengths.append(item.seq_length)
                # print(f"{len(item.to_bytes())} {item.seq_length}")

        # TODO: We can fill this with smiles here if we wish to compute novelty
        self.smiles = []

    @property
    def bytes_per_length(self):
        raise NotImplementedError("This should be implemented by the subclass")

    @property
    def hparams(self):
        return {}

    def __len__(self):
        return len(self.keys)

    @abstractmethod
    def __getitem__(self, item):
        pass


# *** SmolDataset implementations ***


class GeometricDataset(SmolDataset):

    def sample(self, n_items, replacement=False):
        mol_samples = np.random.choice(self._data.to_list(),
                                       n_items,
                                       replace=replacement)
        data = GeometricMolBatch.from_list(mol_samples)
        return GeometricDataset(data, transform=self.transform)

    @classmethod
    def load(cls, data_path, transform=None):
        data = load_smol_data(data_path, GeometricMolBatch)
        data = GeometricMolBatch.from_list(data)
        return GeometricDataset(data, transform=transform)


class LMDBGeometricDataset(LMDBSmolDataset):

    def __getitem__(self, item):
        try:
            key = self.keys[item]
            with self.env.begin() as txn:
                value = txn.get(key.encode('utf-8'))
                mol = GeometricMol.from_bytes(value)
            if self.transform is not None:
                mol = self.transform(mol)
            return mol
        except Exception as e:
            print(f"[error] Skipping idx {item} due to: {e}")
            return self.__getitem__((item + 1) % len(self))  # try next one


class PocketComplexDataset(SmolDataset):

    def sample(self, n_items, replacement=False):
        complex_samples = np.random.choice(self._data.to_list(),
                                           n_items,
                                           replace=replacement)
        data = PocketComplexBatch.from_list(complex_samples)
        return PocketComplexDataset(data, transform=self.transform)

    @classmethod
    def load(cls, data_path, transform=None):
        data = load_smol_data(data_path, PocketComplexBatch)
        data = PocketComplexBatch.from_list(data)
        return PocketComplexDataset(data, transform=transform)


class LMDBPocketComplexDataset(LMDBSmolDataset):

    @property
    def bytes_per_length(self):
        return 0.009

    def __getitem__(self, item):
        try:
            key = self.keys[item]
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode('utf-8'))
            complex = PocketComplex.from_bytes(value)

            if self.transform is not None:
                complex = self.transform(complex)

            if len(complex.holo) + len(complex.ligand) > self.max_length:
                print(
                    f"Skipping idx {item} due to length {len(complex.holo) + len(complex.ligand)} > {self.max_length}"
                )
                return self.__getitem__(np.random.randint(0, len(self)))

            return complex
        except Exception as e:
            print(f"[error] Skipping idx {item} due to: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


class EfficentLMDBPocketComplexDataset(LMDBSmolDataset):

    def setup(self):
        map_size = 100 * 1024**3
        self.protein_lmdb_path = str(Path(self.lmdb_path) / 'protein')
        self.ligand_lmdb_path = str(Path(self.lmdb_path) / 'ligand')

        if os.path.exists(self.new_keys_file):
            self.keys = pickle.load(open(self.new_keys_file, 'rb'))
            return

        os.makedirs(self.protein_lmdb_path, exist_ok=True)
        os.makedirs(self.ligand_lmdb_path, exist_ok=True)

        self.protein_env = lmdb.open(self.protein_lmdb_path,
                                     readonly=False,
                                     lock=False,
                                     map_size=map_size)
        self.ligand_env = lmdb.open(self.ligand_lmdb_path,
                                    readonly=False,
                                    lock=False,
                                    map_size=map_size)

        new_keys = []
        for protein_key in tqdm(self.keys):
            with self.env.begin(write=False) as txn:
                value = txn.get(protein_key.encode('utf-8'))

            raw_bytes = pickle.loads(value)
            protein_bytes = raw_bytes[0]
            ligands_bytes = raw_bytes[1]

            with self.protein_env.begin(write=True) as txn:
                txn.put(protein_key.encode('utf-8'), protein_bytes)

            # check if type is list
            if isinstance(ligands_bytes, list):
                for i, ligand in enumerate(ligands_bytes):
                    ligand_key = protein_key + '_' + str(i)
                    new_keys.append((protein_key, ligand_key))

                    with self.ligand_env.begin(write=True) as txn:
                        txn.put(ligand_key.encode('utf-8'), ligand)
            else:
                new_keys.append((protein_key, protein_key))
                with self.ligand_env.begin(write=True) as txn:
                    txn.put(protein_key.encode('utf-8'), ligands_bytes)

        self.keys = new_keys
        pickle.dump(self.keys, open(self.new_keys_file, 'wb'))

    def __init__(self, key_path, lmdb_path, transform=None):
        super().__init__(key_path, lmdb_path, transform)
        self.new_keys_file = str(Path(lmdb_path) / 'new_keys.pkl')
        self.setup()

        map_size = 100 * 1024**3
        self.protein_env = lmdb.open(self.protein_lmdb_path,
                                     readonly=True,
                                     lock=False,
                                     map_size=map_size)
        self.ligand_env = lmdb.open(self.ligand_lmdb_path,
                                    readonly=True,
                                    lock=False,
                                    map_size=map_size)

    def __getitem__(self, item):
        try:
            key = self.keys[item]
            protein_key, ligand_key = key

            with self.protein_env.begin(write=False) as txn:
                protein_bytes = txn.get(protein_key.encode('utf-8'))
                protein = ProteinPocket.from_bytes(protein_bytes)
            with self.ligand_env.begin(write=False) as txn:
                ligand_bytes = txn.get(ligand_key.encode('utf-8'))
                ligand = GeometricMol.from_bytes(ligand_bytes)

            complex = PocketComplex(holo=protein, ligand=ligand)
            if self.transform is not None:
                complex = self.transform(complex)
            return complex
        except Exception as e:
            print(f"[error] Skipping idx {item} due to: {e}")
            return self.__getitem__((item + 1) % len(self))  # try next one


# *** Other useful datasets ***


class SmolPairDataset(torch.utils.data.Dataset):
    """A dataset which returns pairs of SmolMol objects"""

    def __init__(self, from_dataset: SmolDataset, to_dataset: SmolDataset):
        super().__init__()

        if len(from_dataset) != len(to_dataset):
            raise ValueError(
                "From and to datasets must have the same number of items.")

        if from_dataset.lengths != to_dataset.lengths:
            raise ValueError(
                "From and to datasets must have molecules of the same length at each index."
            )

        self.from_dataset = from_dataset
        self.to_dataset = to_dataset

    # TODO stop hparams clashing from different sources
    @property
    def hparams(self):
        return {**self.from_dataset.hparams, **self.to_dataset.hparams}

    @property
    def lengths(self):
        return self.from_dataset.lengths

    def __len__(self):
        return len(self.from_dataset)

    def __getitem__(self, item):
        from_mol = self.from_dataset[item]
        to_mol = self.to_dataset[item]
        return from_mol, to_mol
