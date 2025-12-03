import h5py
import sys
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from tqdm import tqdm

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from torch_geometric.loader import DataLoader as PyGDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    sys.path.append('/home/catlab/py_code')
    from eg_mamba.modules.connected_components import CustomLargestConnectedComponents
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class DenoisingDataset(Dataset):
    def __init__(self, 
                 hdf5_file_list: List[Path], 
                 specific_dataset_cfg: DictConfig):
        """
        initialize Dataset.

        Args:
            hdf5_file_list (List[Path]): formatted list of HDF5 file paths.
            specific_dataset_cfg (DictConfig): OmegaConf configuration for the specific dataset.
        """
        super().__init__()
        self.hdf5_file_list = hdf5_file_list
        self.specific_cfg = specific_dataset_cfg

        # --- 1. Extract parameters from OmegaConf ---
        try:
            self.graph_params = self.specific_cfg.graph_params

            self.ev_cnt = int(self.specific_cfg.ev_cnt)
            self.overlap_ratio = float(self.specific_cfg.overlap_ratio)

            self.radius = float(self.graph_params.radius)
            self.max_num_neighbors = int(self.graph_params.max_num_neighbors)
            self.loop = bool(self.graph_params.loop)
            
            self.camera_xy = list(self.specific_cfg.camera_xy)
            if len(self.camera_xy) != 2:
                raise ValueError(f"camera_xy must be in [width, height] format, but got {self.camera_xy}")

            self.s_minus_1 = float(max(self.camera_xy[0], self.camera_xy[1]) - 1)

            self.trim_ratio_start = float(self.specific_cfg.get('trim_ratio_start', 0.25))
            self.trim_ratio_end = float(self.specific_cfg.get('trim_ratio_end', 0.15))

            if not (0.0 <= self.trim_ratio_start < 1.0):
                raise ValueError(f"trim_ratio_start ({self.trim_ratio_start}) must be in [0.0, 1.0) range.")
            if not (0.0 <= self.trim_ratio_end < 1.0):
                raise ValueError(f"trim_ratio_end ({self.trim_ratio_end}) must be in [0.0, 1.0) range.")

            num_comp_raw = self.graph_params.num_components
            if num_comp_raw is None or \
               (isinstance(num_comp_raw, str) and num_comp_raw.lower() in ['null', 'none']):
                
                self.num_components = None
            else:
                try:
                    self.num_components = int(num_comp_raw)
                except (ValueError, TypeError):
                    raise ValueError(f"graph_params.num_components: will not convert '{num_comp_raw}' to int.")

            # 2. 处理 min_nodes_lcc
            min_nodes_raw = self.graph_params.min_nodes_lcc
            try:
                # [修复] 强制转换为 int
                self.min_nodes_lcc = int(min_nodes_raw) 
            except (ValueError, TypeError):
                raise ValueError(f"graph_params.min_nodes_lcc: will not convert '{min_nodes_raw}' to int.")

        except AttributeError as e:
            logging.error(f"DenoisingDataset initialization failed: missing required key in .yaml config: {e}")
            raise
        except (TypeError, ValueError) as e:
            logging.error(f"DenoisingDataset initialization failed: parameter type or value error in .yaml: {e}")
            raise

        self.step_size = int(self.ev_cnt * (1.0 - self.overlap_ratio))
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive. Please check ev_cnt ({self.ev_cnt}) and overlap_ratio ({self.overlap_ratio})")

        self.transform = CustomLargestConnectedComponents(
            num_components=self.num_components,
            min_nodes_cnt_in_subgraph=self.min_nodes_lcc,
            min_mean_ratio=self.graph_params.min_mean_ratio
        )
        self.index_map = []
        self._build_index_map()

    def _build_index_map(self):
        total_samples = 0
        
        for h5_path in tqdm(self.hdf5_file_list, desc="Indexing HDF5 files"):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    if 'events/x' not in hf:
                        continue
                        
                    num_events_in_file = hf['events/x'].shape[0]
                    
                    effective_start_idx = round(num_events_in_file * self.trim_ratio_start)
                    effective_end_idx = round(num_events_in_file * (1.0 - self.trim_ratio_end))
                    
                    effective_num_events = effective_end_idx - effective_start_idx

                    if effective_num_events < self.ev_cnt:
                        continue
                        
                    num_samples_in_file = (effective_num_events - self.ev_cnt) // self.step_size + 1
                    
                    for i in range(num_samples_in_file):
                        start_idx = (i * self.step_size) + effective_start_idx
                        end_idx = start_idx + self.ev_cnt
                        
                        if end_idx > effective_end_idx:
                            if end_idx > num_events_in_file:
                                continue
                            pass

                        self.index_map.append( (str(h5_path), start_idx, end_idx) )
                    
                    total_samples += num_samples_in_file
                    
            except (IOError, OSError, KeyError) as e:
                logging.error(f"Cannot process {h5_path.name}: {e}")
        
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Optional[Data]:
        if not (0 <= idx < len(self.index_map)):
            raise IndexError(f"Index {idx} out of range [0, {len(self.index_map) - 1}]")

        h5_path_str, start_idx, end_idx = self.index_map[idx]
        
        try:
            # 1. Open HDF5 file and read slices as needed
            with h5py.File(h5_path_str, 'r') as hf:
                events_group = hf['events']
                x_data = events_group['x'][start_idx:end_idx]
                y_data = events_group['y'][start_idx:end_idx]
                t_data_raw = events_group['timestamp'][start_idx:end_idx]
                p_data = events_group['polarity'][start_idx:end_idx]
                # p_data = events_group['epm'][start_idx:end_idx]
                label_data = events_group['label'][start_idx:end_idx]
                
            # --- 2. Normalization ---

            # a) Normalize X, Y
            x_norm = x_data.astype(np.float32) / self.s_minus_1
            y_norm = y_data.astype(np.float32) / self.s_minus_1

            # b) Normalize T
            t_data_raw = t_data_raw.astype(np.float32)
            t_data_zero_based = t_data_raw - t_data_raw.min()
            
            t_duration = np.maximum(
                t_data_zero_based.max(), 
                np.finfo(np.float32).eps 
            )
            t_norm = t_data_zero_based / t_duration

            # c) Normalize P
            p_data = p_data.astype(np.float32)

            # --- 3. Prepare PyG Attributes ---

            # a) Node Features: (x_norm, y_norm, t_norm, p)
            node_features = torch.from_numpy(
                np.stack([x_norm, y_norm, t_norm, p_data], axis=-1)
            ).float()

            # b) Node Position: (x_norm, y_norm, t_norm)
            pos = torch.from_numpy(
                np.stack([x_norm, y_norm, t_norm], axis=-1)
            ).float()
            orig_coords = pos.clone()
            
            # d) Ground Truth
            denoising_labels = torch.from_numpy(label_data).long()

            # --- 4. Radius Graph ---
            edge_index = radius_graph(
                pos, 
                r=self.radius, 
                batch=None,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors
            )
            
            # --- 5. PyG Data ---
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                pos=pos,
                orig_coords=orig_coords,
                denoising_label=denoising_labels
            )
            
            graph_data = self.transform(graph_data)
            
            if graph_data.num_nodes == 0:
                 return None

            return graph_data

        except (IOError, OSError, KeyError) as e:
            logging.error(f"Failed to read {h5_path_str} (slice {start_idx}:{end_idx}) in __getitem__ (index {idx}): {e}")
            return None
        except Exception as e:
            logging.error(f"Unknown error occurred in __getitem__ (index {idx}): {e}", exc_info=True)
            return None

#################################################################################################
#################################################################################################
#################################################################################################

def load_config(yaml_path: str) -> DictConfig:
    try:
        original_path = Path(yaml_path)
        expanded_path = original_path.expanduser()
        path_to_load = expanded_path
        
        try:
            absolute_path_for_log = path_to_load.resolve(strict=True)
        except FileNotFoundError:
            absolute_path_for_log = path_to_load
        except Exception as resolve_err:
            absolute_path_for_log = path_to_load

        cfg = OmegaConf.load(path_to_load)
        return cfg
    except FileNotFoundError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"There was an error with '{yaml_path}': {e}", exc_info=True)
        raise


class DenoisingDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: DictConfig,
                 dataset_name: str = 'DVSCLEAN'):
        super().__init__()
        self.dataset_config = dataset_config
        self.dataset_name = dataset_name.upper()
        
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: List[Dataset] = []
        self.test_datasets: List[Dataset] = []
        self.predict_datasets: List[Dataset] = []
        
        try:
            base_path_str = self.dataset_config.base_path
            if not base_path_str: 
                raise KeyError
            self.base_path = Path(base_path_str).expanduser()

            self.specific_cfg = self.dataset_config[self.dataset_name]
            if self.specific_cfg is None:
                raise KeyError

            self.dataloader_params = {
                'batch_size_train': self.specific_cfg.train_batch_size,
                'batch_size_val': self.specific_cfg.val_batch_size,
                'batch_size_test': self.specific_cfg.test_batch_size,
                'shuffle_train': self.specific_cfg.train_shuffle,
                'shuffle_val': self.specific_cfg.val_shuffle,
                'shuffle_test': self.specific_cfg.test_shuffle,
                'num_workers': self.dataset_config.num_workers
            }
            
            for key in ['batch_size_train', 'batch_size_val', 'batch_size_test']:
                if self.dataloader_params[key] is None:
                    raise KeyError
            
        except (KeyError, AttributeError, ValueError) as e:
            logging.error(f"DataModule __init__ failed: Configuration error - {e}", exc_info=True)
            raise

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset or self.val_datasets or self.test_datasets:
            return

        dataset_dir = self.base_path / self.dataset_name / "hdf5_dataset"
        if not dataset_dir.exists():
            raise FileNotFoundError

        train_files_list = []
        val_files_list_of_lists = []
        test_files_list_of_lists = []
        
        # 1. dataset_name specific setup
        if self.dataset_name == "DVSCLEAN":
            train_mixed_dir = dataset_dir / "train"
            train_50_dir = dataset_dir / "train_50"
            train_100_dir = dataset_dir / "train_100"
            val_50_dir = dataset_dir / "val_50"
            val_100_dir = dataset_dir / "val_100"
            
            train_type = self.specific_cfg.get('train_type', 'train_dvsclean')
            if train_type == 'train_50':
                train_files_list = sorted(list(train_50_dir.glob("*.hdf5")))
            elif train_type == 'train_100':
                train_files_list = sorted(list(train_100_dir.glob("*.hdf5")))
            elif train_type == 'train_dvsclean':
                train_files_list = sorted(list(train_mixed_dir.glob("*.hdf5")))
            else:
                raise ValueError

            val_type = self.specific_cfg.get('val_type', 'val_dvsclean')
            if val_type == 'val_50':
                val_files_list_of_lists.append(sorted(list(val_50_dir.glob("*.hdf5"))))
            elif val_type == 'val_100':
                val_files_list_of_lists.append(sorted(list(val_100_dir.glob("*.hdf5"))))
            elif val_type == 'val_dvsclean':
                val_files_list_of_lists.append(sorted(list(val_50_dir.glob("*.hdf5"))))
                val_files_list_of_lists.append(sorted(list(val_100_dir.glob("*.hdf5"))))
            else:
                raise ValueError
            
            test_files_list_of_lists = val_files_list_of_lists

        elif self.dataset_name == "EDKOGTL":
            train_dir = dataset_dir / "train"
            val_dir = dataset_dir / "val"
            train_good_file = train_dir / "goodlight_750lux_train.hdf5"
            train_low_file = train_dir / "lowlight_5lux_train.hdf5"
            val_good_file = val_dir / "goodlight_750lux_val.hdf5"
            val_low_file = val_dir / "lowlight_5lux_val.hdf5"

            train_type = self.specific_cfg.get('train_type', 'train_edkogtl')
            if train_type == 'train_good':
                train_files_list = [train_good_file]
            elif train_type == 'train_low':
                train_files_list = [train_low_file]
            elif train_type == 'train_edkogtl':
                train_files_list = [train_good_file, train_low_file]
            else:
                raise ValueError

            # --- c. 解析验证类型 ---
            val_type = self.specific_cfg.get('val_type', 'val_edkogtl')
            if val_type == 'val_good':
                val_files_list_of_lists.append([val_good_file])
            elif val_type == 'val_low':
                val_files_list_of_lists.append([val_low_file])
            elif val_type == 'val_edkogtl':
                val_files_list_of_lists.append([val_good_file])
                val_files_list_of_lists.append([val_low_file])
            else:
                raise ValueError

            test_files_list_of_lists = val_files_list_of_lists

        else:
            raise ValueError

        
        if stage == "fit" or stage is None:
            self.train_dataset = DenoisingDataset(train_files_list, self.specific_cfg)
            
            self.val_datasets = []
            for i, files in enumerate(val_files_list_of_lists):
                self.val_datasets.append(DenoisingDataset(files, self.specific_cfg))
            
        if stage == "test" or stage is None:
            self.test_datasets = []
            for i, files in enumerate(test_files_list_of_lists):
                self.test_datasets.append(DenoisingDataset(files, self.specific_cfg))
                
        self.predict_datasets = self.test_datasets

    def _create_dataloader(self, 
                           dataset: Optional[Dataset], 
                           batch_size: int, 
                           shuffle: bool) -> Optional[PyGDataLoader]:
        if not dataset or len(dataset) == 0:
            return None

        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_params['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.dataloader_params['num_workers'] > 0 else False,
            drop_last=True
        )

    def train_dataloader(self) -> PyGDataLoader:
        loader = self._create_dataloader(
            self.train_dataset,
            self.dataloader_params['batch_size_train'],
            self.dataloader_params['shuffle_train']
        )
        if loader is None:
            logging.error("Train DataLoader is None!")
        return loader

    def val_dataloader(self) -> List[PyGDataLoader]:
        loaders = []
        for i, dataset in enumerate(self.val_datasets):
            if dataset and len(dataset) > 0:
                loader = self._create_dataloader(
                    dataset,
                    self.dataloader_params['batch_size_val'],
                    shuffle=self.dataloader_params.get('shuffle_val', False)
                )
                if loader:
                    loaders.append(loader)
            else:
                logging.warning(f"Validation dataset at index {i} is empty or None.")
        
        if not loaders:
            logging.error("Validation DataLoader list is empty!")
        
        return loaders

    def test_dataloader(self) -> List[PyGDataLoader]:
        loaders = []
        for i, dataset in enumerate(self.test_datasets):
            if dataset and len(dataset) > 0:
                loader = self._create_dataloader(
                    dataset,
                    self.dataloader_params['batch_size_test'],
                    shuffle=False
                )
                if loader:
                    loaders.append(loader)
            else:
                logging.warning(f"Test dataset at index {i} is empty or None.")
        
        if not loaders:
            logging.error("Test DataLoader list is empty!")

        return loaders

    def predict_dataloader(self) -> List[PyGDataLoader]:
        if not self.test_datasets and self.predict_datasets:
             return self.predict_datasets # Fallback
        if not self.test_datasets and not self.predict_datasets:
            self.setup('test')
        self.predict_datasets = self.test_datasets
        return self.predict_datasets

