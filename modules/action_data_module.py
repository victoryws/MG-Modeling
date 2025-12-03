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
    logging.error("Failed to import CustomLargestConnectedComponents. Please ensure that the eg_mamba.modules package is accessible.")
    sys.exit(1)


#################################################################################################
# [general] ActionDataset class
#################################################################################################

class ActionDataset(Dataset):
    """
    [!] General Action Recognition Dataset (Compatible with DVSACTION, THUEACT50CHL, etc.)

    Dynamically creates graph samples by applying a sliding window on HDF5 files.
    """
    def __init__(self, 
                 hdf5_file_list: List[Path], 
                 specific_dataset_cfg: DictConfig):
        """
        Initialize Dataset.

        Args:
            hdf5_file_list (List[Path]): List of paths to all HDF5 files for this dataset (e.g., 'train').
            specific_dataset_cfg (DictConfig): Specific dataset configuration loaded from YAML (e.g., cfg.dataset.DVSACTION).
        """
        super().__init__()
        self.hdf5_file_list = hdf5_file_list
        self.specific_cfg = specific_dataset_cfg

        # --- 1. Extract parameters from OmegaConf configuration ---
        try:
            self.graph_params = self.specific_cfg.graph_params
            self.ev_cnt = int(self.specific_cfg.ev_cnt)
            self.overlap_ratio = float(self.specific_cfg.overlap_ratio)
            self.radius = float(self.graph_params.radius)
            self.max_num_neighbors = int(self.graph_params.max_num_neighbors)
            self.loop = bool(self.graph_params.loop)

            self.camera_xy = list(self.specific_cfg.camera_xy)
            if len(self.camera_xy) != 2:
                raise ValueError(f"camera_xy must be in [width, height] format, got {self.camera_xy}")

            self.s_minus_1 = float(max(self.camera_xy[0], self.camera_xy[1]) - 1)
            logging.info(f"ActionDataset: Normalized s_minus_1 set to {self.s_minus_1} (based on camera_xy: {self.camera_xy})")

            self.trim_ratio_start = float(self.specific_cfg.get('trim_ratio_start', 0.25))
            self.trim_ratio_end = float(self.specific_cfg.get('trim_ratio_end', 0.15))
            
            if not (0.0 <= self.trim_ratio_start < 1.0):
                raise ValueError(f"trim_ratio_start ({self.trim_ratio_start}) must be in [0.0, 1.0) range.")
            if not (0.0 <= self.trim_ratio_end < 1.0):
                raise ValueError(f"trim_ratio_end ({self.trim_ratio_end}) must be in [0.0, 1.0) range.")
            if (self.trim_ratio_start + self.trim_ratio_end) >= 1.0:
                raise ValueError(f"trim_ratio_start and trim_ratio_end sum ({self.trim_ratio_start + self.trim_ratio_end}) must be less than 1.0.")

            num_comp_raw = self.graph_params.num_components
            if num_comp_raw is None or \
               (isinstance(num_comp_raw, str) and num_comp_raw.lower() in ['null', 'none']):
                self.num_components = None
            else:
                self.num_components = int(num_comp_raw)

            min_nodes_raw = self.graph_params.min_nodes_lcc
            self.min_nodes_lcc = int(min_nodes_raw) 
            
        except (AttributeError, KeyError) as e:
            logging.error(f"ActionDataset initialization failed: Missing required keys in .yaml configuration: {e}")
            raise
        except (TypeError, ValueError) as e:
            logging.error(f"ActionDataset initialization failed: Type or value error in .yaml parameters: {e}")
            raise

        # --- 2. Compute sliding window parameters ---
        self.step_size = int(self.ev_cnt * (1.0 - self.overlap_ratio))
        if self.step_size <= 0:
            raise ValueError(f"Step size (step_size) must be positive. Please check ev_cnt ({self.ev_cnt}) and overlap_ratio ({self.overlap_ratio})")

        # --- 3. Instantiate transformations ---
        try:
            self.transform = CustomLargestConnectedComponents(
                num_components=self.num_components,
                min_nodes_cnt_in_subgraph=self.min_nodes_lcc
            )
            logging.info(f"ActionDataset: Initialized LCC transformation (num_comp={self.num_components}, min_nodes={self.min_nodes_lcc})")
        except Exception as e:
            logging.error(f"Failed to instantiate CustomLargestConnectedComponents: {e}")
            raise

        # --- 4. Build index map ---
        self.index_map = []
        self._build_index_map()

    def _build_index_map(self):
        """ 
        [Modified] Traverse all HDF5 files and precompute all valid sliding window slices.

        """
        total_samples = 0
        
        for h5_path in tqdm(self.hdf5_file_list, desc="Indexing HDF5 files"):
            try:
                with h5py.File(h5_path, 'r') as hf:
                    # [!] 依赖于 "event_N4" 键
                    if 'event_N4' not in hf:
                        continue
                        
                    num_events_in_file = hf['event_N4'].shape[0]
                    
                    effective_start_idx = round(num_events_in_file * self.trim_ratio_start)
                    effective_end_idx = round(num_events_in_file * (1.0 - self.trim_ratio_end))
                    effective_num_events = effective_end_idx - effective_start_idx

                    if effective_num_events < self.ev_cnt:
                        continue
                        
                    num_samples_in_file = (effective_num_events - self.ev_cnt) // self.step_size + 1
                    
                    # 5. [修改] 遍历有效区域
                    for i in range(num_samples_in_file):
                        start_idx = (i * self.step_size) + effective_start_idx
                        end_idx = start_idx + self.ev_cnt
                        
                        if end_idx > num_events_in_file:
                            continue
                        
                        self.index_map.append( (str(h5_path), start_idx, end_idx) )
                    
                    total_samples += num_samples_in_file
                        
            except (IOError, OSError, KeyError) as e:
                logging.error(f"Failed to process file {h5_path.name}: {e}")

        if not self.index_map:
            logging.critical("Index map is empty! No valid samples found.")
        else:
            logging.info(f"Index map construction complete. Total Samples: {total_samples:,}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Optional[Data]:
        if not (0 <= idx < len(self.index_map)):
            raise IndexError(f"Index {idx} out of range [0, {len(self.index_map) - 1}]")

        h5_path_str, start_idx, end_idx = self.index_map[idx]
        
        try:
            with h5py.File(h5_path_str, 'r') as hf:
                event_data_slice = hf['event_N4'][start_idx:end_idx, :]
                label_data = hf['label'][()]

            x_data = event_data_slice[:, 0]
            y_data = event_data_slice[:, 1]
            t_data_raw = event_data_slice[:, 2]
            p_data = event_data_slice[:, 3]
            
            x_norm = x_data.astype(np.float32) / self.s_minus_1
            y_norm = y_data.astype(np.float32) / self.s_minus_1
            
            t_data_raw = t_data_raw.astype(np.float32)
            t_data_zero_based = t_data_raw - t_data_raw.min()
            t_duration = np.maximum(t_data_zero_based.max(), np.finfo(np.float32).eps)
            t_norm = t_data_zero_based / t_duration
            
            p_data = p_data.astype(np.float32)

            node_features = torch.from_numpy(
                np.stack([x_norm, y_norm, t_norm, p_data], axis=-1)
            ).float()
            
            pos = torch.from_numpy(
                np.stack([x_norm, y_norm, t_norm], axis=-1)
            ).float()
            
            orig_coords = pos.clone()
            action_label_tensor = torch.tensor([label_data]).long()

            edge_index = radius_graph(
                pos, 
                r=self.radius, 
                batch=None,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors
            )
            
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                pos=pos,
                orig_coords=orig_coords,
                action_label=action_label_tensor # [!] 您的命名要求
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



#################################################################################################
# [general] ActionDataModule class
#################################################################################################

class ActionDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_config: DictConfig,
                 dataset_name: str):
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
                raise KeyError("Config 'base_path' is missing or empty.")
            self.base_path = Path(base_path_str).expanduser()
            
            num_workers_val = self.dataset_config.num_workers
            if num_workers_val is None:
                raise KeyError("Config 'num_workers' is missing.")

            self.specific_cfg = self.dataset_config[self.dataset_name]
            if self.specific_cfg is None:
                raise KeyError(f"Config for dataset '{self.dataset_name}' not found.")

            self.dataloader_params = {
                'batch_size_train': self.specific_cfg.train_batch_size,
                'batch_size_val': self.specific_cfg.val_batch_size,
                'batch_size_test': self.specific_cfg.test_batch_size,
                'shuffle_train': self.specific_cfg.train_shuffle,
                'shuffle_val': self.specific_cfg.val_shuffle,
                'shuffle_test': self.specific_cfg.test_shuffle,
                'num_workers': int(num_workers_val)
            }
            
            for key, value in self.dataloader_params.items():
                if value is None:
                    raise KeyError(f"Config '{self.dataset_name}' or top-level is missing parameter: '{key}'")

            logging.info(f"ActionDataModule for '{self.dataset_name}' initialized successfully.")
            logging.info(f"  Base Path: {self.base_path}")
            logging.info(f"  Num Workers: {self.dataloader_params['num_workers']}")

        except (KeyError, AttributeError, ValueError) as e:
            logging.error(f"DataModule __init__ failed: Configuration error - {e}", exc_info=True)
            raise

    def setup(self, stage: Optional[str] = None):
        """ [general] Setup dataset (automatically called by PTL). """

        if self.train_dataset or self.val_datasets or self.test_datasets:
            logging.info("DataModule setup has already been called, skipping.")
            return

        logging.info(f"DataModule setup ( {self.dataset_name} ) for stage: {stage}")

        dataset_dir = self.base_path / self.dataset_name / "hdf5_dataset"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"Missing 'train' (at {train_dir}) or 'val' (at {val_dir}) folder.")

        train_files_list = sorted(list(train_dir.glob("*.hdf5")))

        val_files_list = sorted(list(val_dir.glob("*.hdf5")))
        test_files_list = val_files_list
        
        if stage == "fit" or stage is None:
            if not train_files_list: logging.error(f"'fit' stage found no training files in {train_dir}.")
            self.train_dataset = ActionDataset(train_files_list, self.specific_cfg)

            if not val_files_list: logging.warning(f"'fit' stage found no validation files in {val_dir}.")
            self.val_datasets = [ActionDataset(val_files_list, self.specific_cfg)]
            
            logging.info(f"{self.dataset_name} 'fit' setup: {len(self.train_dataset)} train samples.")
            val_lengths = [len(ds) for ds in self.val_datasets]
            logging.info(f"  (val): {len(self.val_datasets)} Val datasets (samples: {val_lengths})")

        if stage == "test" or stage is None:
            if not test_files_list: logging.warning(f"'test' stage found no testing files in {val_dir}.")
            self.test_datasets = [ActionDataset(test_files_list, self.specific_cfg)]
            
            test_lengths = [len(ds) for ds in self.test_datasets]
            logging.info(f"{self.dataset_name} 'test' setup: {len(self.test_datasets)} Test datasets (samples: {test_lengths})")
        
        if stage == "predict":
            if not test_files_list: logging.warning(f"'predict' stage found no prediction files in {val_dir}.")
            self.predict_datasets = [ActionDataset(test_files_list, self.specific_cfg)]


    def _create_dataloader(self, 
                           dataset: Optional[Dataset], 
                           batch_size: int, 
                           shuffle: bool) -> Optional[PyGDataLoader]:
        """ [general] Internal helper function """
        if not dataset or len(dataset) == 0:
            logging.warning(f"Dataset is empty, cannot create DataLoader.")
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
        logging.info(f"Creating train_dataloader (batch_size={self.dataloader_params['batch_size_train']})...")
        loader = self._create_dataloader(
            self.train_dataset,
            self.dataloader_params['batch_size_train'],
            self.dataloader_params['shuffle_train']
        )
        if loader is None:
            logging.error("Train DataLoader is empty!")
        return loader

    def val_dataloader(self) -> List[PyGDataLoader]:
        logging.info(f"Creating val_dataloader list (batch_size={self.dataloader_params['batch_size_val']})...")
        loaders = []
        for i, dataset in enumerate(self.val_datasets):
            if dataset and len(dataset) > 0:
                loader = self._create_dataloader(
                    dataset,
                    self.dataloader_params['batch_size_val'],
                    shuffle=self.dataloader_params['shuffle_val']
                )
                if loader:
                    loaders.append(loader)
        if not loaders: logging.error("Validation DataLoader list is empty!")
        return loaders

    def test_dataloader(self) -> List[PyGDataLoader]:
        logging.info(f"Creating test_dataloader list (batch_size={self.dataloader_params['batch_size_test']})...")
        loaders = []
        for i, dataset in enumerate(self.test_datasets):
            if dataset and len(dataset) > 0:
                loader = self._create_dataloader(
                    dataset,
                    self.dataloader_params['batch_size_test'],
                    shuffle=self.dataloader_params['shuffle_test']
                )
                if loader:
                    loaders.append(loader)
        if not loaders: logging.error("Test DataLoader list is empty!")
        return loaders

    def predict_dataloader(self) -> List[PyGDataLoader]:
        logging.info("Using test_dataloaders for prediction.")
        if not self.predict_datasets:
            self.setup('predict')
        
        loaders = []
        for i, dataset in enumerate(self.predict_datasets):
             if dataset and len(dataset) > 0:
                loader = self._create_dataloader(
                    dataset,
                    self.dataloader_params['batch_size_test'],
                    shuffle=False
                )
                if loader:
                    loaders.append(loader)
        return loaders


