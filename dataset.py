"""
Dataset Classes for 3D Body Scan to DXA Prediction
Handles Normal, Bariatric, and Sarcopenia datasets
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.scan import preprocess_scan
from utils.dxa_image import preprocess_dxa_image
from utils.pdf_parser import DXACompositionParser


class BodyScanDXADataset(Dataset):
    """
    Base dataset class for 3D body scan + DXA + composition
    
    Returns:
        scan: (N, 3) point cloud
        dxa_image: (H, W) DXA image
        composition: (M,) body composition vector
        metadata: Dictionary with additional info
    """
    
    def __init__(self,
                 scan_paths: List[str],
                 dxa_paths: List[str],
                 composition_data: pd.DataFrame,
                 population: str,
                 n_vertices: int = 100000,
                 dxa_size: Tuple[int, int] = (320, 864),
                 augment: bool = False,
                 augment_params_3d: Optional[Dict] = None,
                 augment_params_dxa: Optional[Dict] = None,
                 seed: int = 0):
        """
        Args:
            scan_paths: List of 3D scan file paths
            dxa_paths: List of DXA image paths (same order as scans)
            composition_data: DataFrame with body composition measurements
            population: 'normal', 'bariatric', or 'sarcopenia'
            n_vertices: Target number of vertices
            dxa_size: (width, height) for DXA images
            augment: Whether to apply augmentation
            augment_params_3d: Parameters for 3D augmentation
            augment_params_dxa: Parameters for DXA augmentation
            seed: Random seed
        """
        self.scan_paths = scan_paths
        self.dxa_paths = dxa_paths
        self.composition_data = composition_data
        self.population = population
        self.n_vertices = n_vertices
        self.dxa_size = dxa_size
        self.augment = augment
        self.augment_params_3d = augment_params_3d or {}
        self.augment_params_dxa = augment_params_dxa or {}
        self.seed = seed
        
        # Verify lengths match
        assert len(scan_paths) == len(dxa_paths), \
            f"Mismatch: {len(scan_paths)} scans vs {len(dxa_paths)} DXA images"
        
        # Extract composition measurements into numpy array
        # Filter to only numeric columns (exclude metadata like device names)
        exclude_cols = ['subject_id', 'pdf_path', 'scan_date', 'patient_id', 'sex', 'ethnicity',
                        'height_cm', 'weight_kg', 'age', 'bmi']  # Extract these separately as anthropometric
        
        self.composition_fields = []
        for col in composition_data.columns:
            if col in exclude_cols:
                continue
            # Try to convert to numeric, skip if it fails
            try:
                pd.to_numeric(composition_data[col], errors='raise')
                self.composition_fields.append(col)
            except (ValueError, TypeError):
                continue
        
        self.composition_array = composition_data[self.composition_fields].values.astype(np.float32)
        
        # Handle missing/invalid values to avoid NaNs in loss computation
        if np.isnan(self.composition_array).any() or np.isinf(self.composition_array).any():
            # Replace inf with NaN for unified handling
            self.composition_array[~np.isfinite(self.composition_array)] = np.nan
            col_means = np.nanmean(self.composition_array, axis=0)
            # If a column is entirely NaN, fill with 0.0
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            nan_mask = np.isnan(self.composition_array)
            self.composition_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        print(f"Dataset initialized: {len(self)} samples, {len(self.composition_fields)} composition measurements")
    
    def __len__(self):
        return len(self.scan_paths)
    
    def __getitem__(self, idx):
        # Random seed for this sample (for reproducibility)
        sample_seed = self.seed + idx if not self.augment else np.random.randint(0, 1000000)
        
        # Load and preprocess 3D scan
        scan_path = self.scan_paths[idx]
        scan, scan_meta = preprocess_scan(
            scan_path,
            target_vertices=self.n_vertices,
            normalization='unit_sphere',
            augment=self.augment,
            augment_params=self.augment_params_3d if self.augment else None,
            seed=sample_seed
        )
        
        # Load and preprocess DXA image
        dxa_path = self.dxa_paths[idx]
        dxa_image, dxa_mask, dxa_meta = preprocess_dxa_image(
            dxa_path,
            dataset_type=self.population,
            target_size=self.dxa_size,
            normalization='zscore_per_image',  # Use per-image normalization for downstream
            create_mask=True,
            augment=self.augment,
            augment_params=self.augment_params_dxa if self.augment else None,
            seed=sample_seed
        )
        
        # Get composition measurements
        composition = self.composition_array[idx]
        
        # Extract anthropometric features (height, weight, age, sex)
        # These preserve absolute scale information lost in normalization
        row = self.composition_data.iloc[idx]
        
        def _get(val, default=np.nan):
            try:
                f = float(val)
                return f if np.isfinite(f) else default
            except Exception:
                return default

        height_cm = _get(row.get('height_cm', np.nan), 170.0)
        weight_kg = _get(row.get('weight_kg', np.nan), 70.0)
        age_val = _get(row.get('age', np.nan), 50.0)

        # Sex: encode as binary (0=Female, 1=Male)
        sex_raw = row.get('sex', None)
        if isinstance(sex_raw, str):
            sex_encoded = 1.0 if sex_raw.strip().lower().startswith('m') else 0.0
        else:
            try:
                sex_encoded = 1.0 if float(sex_raw) > 0.5 else 0.0
            except Exception:
                sex_encoded = 0.5

        h_m = height_cm / 100.0
        bmi_val = weight_kg / (h_m * h_m) if h_m > 0 else 25.0

        # Ethnicity one-hot (White, Black, Asian, Mixed, Other)
        eth_raw = row.get('ethnicity', '')
        eth_raw = str(eth_raw).lower() if eth_raw is not None else ''
        eth_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0]
        if 'white' in eth_raw or 'british' in eth_raw:
            eth_one_hot[0] = 1.0
        elif 'black' in eth_raw or 'caribbean' in eth_raw or 'african' in eth_raw:
            eth_one_hot[1] = 1.0
        elif 'asian' in eth_raw or 'indian' in eth_raw or 'pakistani' in eth_raw or 'chinese' in eth_raw:
            eth_one_hot[2] = 1.0
        elif 'mixed' in eth_raw:
            eth_one_hot[3] = 1.0
        else:
            eth_one_hot[4] = 1.0

        anthropometric_features = [
            height_cm, weight_kg, age_val, sex_encoded, bmi_val,
            eth_one_hot[0], eth_one_hot[1], eth_one_hot[2], eth_one_hot[3], eth_one_hot[4]
        ]
        anthropometric_tensor = torch.tensor(anthropometric_features, dtype=torch.float32)
        
        # Convert to tensors
        scan_tensor = torch.from_numpy(scan).float()  # (N, 3)
        dxa_tensor = torch.from_numpy(dxa_image).float().unsqueeze(0)  # (1, H, W)
        composition_tensor = torch.from_numpy(composition).float()  # (M,)
        
        # Optional: mask tensor
        if dxa_mask is not None:
            mask_tensor = torch.from_numpy(dxa_mask / 255.0).float().unsqueeze(0)  # (1, H, W)
        else:
            mask_tensor = torch.ones_like(dxa_tensor)
        
        # Metadata
        metadata = {
            'scan_path': str(scan_path),
            'dxa_path': str(dxa_path),
            'population': self.population,
            'index': idx,
            **scan_meta,
            **dxa_meta,
        }
        
        return {
            'scan': scan_tensor,
            'dxa_image': dxa_tensor,
            'dxa_mask': mask_tensor,
            'composition': composition_tensor,
            'anthropometric': anthropometric_tensor,  # Height, weight, age, sex, scale
            'metadata': metadata
        }


class NormalDataset(BodyScanDXADataset):
    """
    Dataset for normal population
    
    Structure:
    - scans/no/{subject_id}_{scan_num}.ply
    - dxa/ba/{subject_id}_1/composition.pdf
    - dxa/ba/{subject_id}_1/c.png
    
    Strategy: Use all scans for training, first scan for testing
    """
    
    @classmethod
    def from_directory(cls,
                       data_root: str,
                       mode: str = 'train',
                       use_all_scans: bool = True,
                       **kwargs):
        """
        Create dataset from directory structure
        
        Args:
            data_root: Root data directory
            mode: 'train', 'val', or 'test'
            use_all_scans: If True, use all scans; if False, use first scan only
            **kwargs: Additional arguments for BodyScanDXADataset
        """
        data_root = Path(data_root)
        scans_dir = data_root / "scans" / "no"
        dxa_dir = data_root / "dxa" / "no"
        
        # Parse composition PDFs first (these are unique per subject)
        parser = DXACompositionParser(convert_to_metric=True)
        composition_df = parser.parse_directory(str(dxa_dir), pattern="*/composition.pdf")
        
        # Find all scan files
        scan_files = list(scans_dir.glob("*.ply"))
        
        # Group scans by subject ID
        subject_scans = {}
        for scan_file in scan_files:
            # Extract subject ID from filename (e.g., "109_1.ply" -> subject "109")
            subject_id = scan_file.stem.split('_')[0]
            if subject_id not in subject_scans:
                subject_scans[subject_id] = []
            subject_scans[subject_id].append(scan_file)
        
        # Sort each subject's scans by scan number
        for subject_id in subject_scans:
            subject_scans[subject_id] = sorted(
                subject_scans[subject_id], 
                key=lambda x: int(x.stem.split('_')[1])
            )
        
        # Iterate over DXA reports (unique subjects) and match to scans
        scan_paths = []
        dxa_paths = []
        composition_rows = []
        
        for _, comp_row in composition_df.iterrows():
            subject_id = comp_row['subject_id']
            
            # Check if DXA image exists
            dxa_folder = dxa_dir / subject_id
            dxa_image = dxa_folder / "t.png"
            
            if not dxa_image.exists():
                continue
            
            # Find scan for this subject
            if subject_id not in subject_scans:
                continue
            
            # Use first scan only (to avoid duplicates from same person)
            scan_file = subject_scans[subject_id][0]
            
            # Add to lists
            scan_paths.append(str(scan_file))
            dxa_paths.append(str(dxa_image))
            composition_rows.append(comp_row)
        
        # Create composition DataFrame
        composition_df_filtered = pd.DataFrame(composition_rows).reset_index(drop=True)
        
        print(f"Normal dataset ({mode}): {len(scan_paths)} samples from {len(composition_df)} subjects")
        
        return cls(
            scan_paths=scan_paths,
            dxa_paths=dxa_paths,
            composition_data=composition_df_filtered,
            population='normal',
            **kwargs
        )


class BariatricDataset(BodyScanDXADataset):
    """
    Dataset for bariatric population
    
    Structure:
    - scans/ba/{subject_id}_{visit}_{scan}.ply
    - dxa/ba/{subject_id}_{visit}/composition.pdf
    - dxa/ba/{subject_id}_{visit}/c.png
    
    Strategy: Use only visits with DXA (visit 1 or 5)
    """
    
    @classmethod
    def from_directory(cls,
                       data_root: str,
                       mode: str = 'train',
                       use_all_scans: bool = True,
                       **kwargs):
        """Create dataset from directory structure"""
        data_root = Path(data_root)
        scans_dir = data_root / "scans" / "ba"
        dxa_dir = data_root / "dxa" / "ba"
        
        # Parse composition PDFs first (unique per subject-visit)
        parser = DXACompositionParser(convert_to_metric=True)
        composition_df = parser.parse_directory(str(dxa_dir), pattern="*/composition.pdf")
        
        # Find all scan files
        scan_files = list(scans_dir.glob("*.ply"))
        
        # Group scans by subject_visit key
        subject_visit_scans = {}
        for scan_file in scan_files:
            # Extract subject and visit (e.g., "1_2_1.ply" -> "1_2")
            parts = scan_file.stem.split('_')
            if len(parts) >= 3:
                visit_key = f"{parts[0]}_{parts[1]}"
                
                if visit_key not in subject_visit_scans:
                    subject_visit_scans[visit_key] = []
                subject_visit_scans[visit_key].append(scan_file)
        
        # Sort each visit's scans by scan number
        for visit_key in subject_visit_scans:
            subject_visit_scans[visit_key] = sorted(
                subject_visit_scans[visit_key],
                key=lambda x: int(x.stem.split('_')[2]) if len(x.stem.split('_')) >= 3 else 0
            )
        
        # Iterate over DXA reports (unique subject-visits) and match to scans
        scan_paths = []
        dxa_paths = []
        composition_rows = []
        
        for _, comp_row in composition_df.iterrows():
            visit_key = comp_row['subject_id']  # e.g., "10_1"
            
            # Check if DXA image exists
            dxa_folder = dxa_dir / visit_key
            dxa_image = dxa_folder / "t.png"
            
            if not dxa_image.exists():
                continue
            
            # Find scans for this visit
            if visit_key not in subject_visit_scans:
                continue
            
            # Use first scan only (to avoid duplicates from same visit)
            scan_file = subject_visit_scans[visit_key][0]
            
            # Add to lists
            scan_paths.append(str(scan_file))
            dxa_paths.append(str(dxa_image))
            composition_rows.append(comp_row)
        
        composition_df_filtered = pd.DataFrame(composition_rows).reset_index(drop=True)
        
        print(f"Bariatric dataset ({mode}): {len(scan_paths)} samples from {len(composition_df)} visits")
        
        return cls(
            scan_paths=scan_paths,
            dxa_paths=dxa_paths,
            composition_data=composition_df_filtered,
            population='bariatric',
            **kwargs
        )


class SarcopeniaDataset(BodyScanDXADataset):
    """
    Dataset for sarcopenia population
    
    Structure:
    - scans/Sarcopenia/OIM study/OIM-{id}/*.obj
    - scans/Sarcopenia/OIM study/OIM-{id}/*_2.jpg (DXA image)
    - scans/Sarcopenia/OIM study/OIM-{id}/*.pdf (composition)
    """
    
    @classmethod
    def from_directory(cls,
                       data_root: str,
                       mode: str = 'train',
                       **kwargs):
        """Create dataset from directory structure"""
        data_root = Path(data_root)
        scans_root = data_root / "scans" / "Sarcopenia" / "OIM study"
        
        # Find subject folders
        subject_folders = [f for f in scans_root.iterdir() if f.is_dir() and f.name.startswith('OIM-')]
        
        # Parse compositions
        parser = DXACompositionParser(convert_to_metric=True)
        
        scan_paths = []
        dxa_paths = []
        composition_list = []
        
        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            
            # Find .obj file
            obj_files = list(subject_folder.glob("*.obj"))
            if len(obj_files) == 0:
                continue
            scan_file = obj_files[0]
            
            # Find DXA image (*_2.jpg)
            dxa_files = list(subject_folder.glob("*_2.jpg")) + list(subject_folder.glob("*_2.png"))
            if len(dxa_files) == 0:
                continue
            dxa_file = dxa_files[0]
            
            # Find PDF
            pdf_files = list(subject_folder.glob("*.pdf"))
            if len(pdf_files) == 0:
                continue
            pdf_file = pdf_files[0]
            
            # Parse composition
            try:
                composition = parser.parse_pdf(str(pdf_file))
                composition['subject_id'] = subject_id
                
                scan_paths.append(str(scan_file))
                dxa_paths.append(str(dxa_file))
                composition_list.append(composition)
            except Exception as e:
                warnings.warn(f"Failed to process {subject_id}: {e}")
                continue
        
        composition_df = pd.DataFrame(composition_list)
        
        print(f"Sarcopenia dataset ({mode}): {len(scan_paths)} samples")
        
        return cls(
            scan_paths=scan_paths,
            dxa_paths=dxa_paths,
            composition_data=composition_df,
            population='sarcopenia',
            **kwargs
        )


def create_combined_dataset(data_root: str,
                            mode: str = 'train',
                            populations: List[str] = ['normal', 'bariatric', 'sarcopenia'],
                            **kwargs):
    """
    Create combined dataset from multiple populations
    
    Args:
        data_root: Root data directory
        mode: 'train', 'val', or 'test'
        populations: Which populations to include
        **kwargs: Additional arguments for datasets
    
    Returns:
        Combined dataset
    """
    from torch.utils.data import ConcatDataset
    
    datasets = []
    
    if 'normal' in populations:
        ds_normal = NormalDataset.from_directory(data_root, mode=mode, **kwargs)
        datasets.append(ds_normal)
    
    if 'bariatric' in populations:
        ds_bariatric = BariatricDataset.from_directory(data_root, mode=mode, **kwargs)
        datasets.append(ds_bariatric)
    
    if 'sarcopenia' in populations:
        ds_sarcopenia = SarcopeniaDataset.from_directory(data_root, mode=mode, **kwargs)
        datasets.append(ds_sarcopenia)
    
    if len(datasets) == 0:
        raise ValueError(f"No datasets created for populations: {populations}")
    
    # Align composition fields across all datasets
    # Use intersection of fields (only fields that exist in ALL datasets)
    if len(datasets) > 1:
        # Get intersection of composition fields
        common_fields = set(datasets[0].composition_fields)
        for ds in datasets[1:]:
            common_fields = common_fields.intersection(set(ds.composition_fields))
        common_fields = sorted(list(common_fields))
        
        print(f"\nAligning composition fields: {len(common_fields)} common fields across {len(datasets)} datasets")
        
        # Update each dataset to use only common fields
        for ds in datasets:
            # Get indices of common fields in this dataset
            field_indices = [ds.composition_fields.index(field) for field in common_fields]
            
            # Update composition array and fields
            ds.composition_array = ds.composition_array[:, field_indices]
            ds.composition_fields = common_fields
            
            print(f"  {ds.population}: {len(ds)} samples, {len(ds.composition_fields)} fields")
    
    combined = ConcatDataset(datasets)
    
    print(f"\nCombined dataset ({mode}): {len(combined)} total samples")
    
    return combined


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset classes...")
    
    DATA_ROOT = "D:/human"
    
    # Test normal dataset
    print("\n1. Testing Normal Dataset:")
    ds_normal = NormalDataset.from_directory(
        DATA_ROOT,
        mode='train',
        use_all_scans=True,
        n_vertices=10000,  # Small for testing
        augment=False
    )
    
    if len(ds_normal) > 0:
        sample = ds_normal[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Scan shape: {sample['scan'].shape}")
        print(f"  DXA shape: {sample['dxa_image'].shape}")
        print(f"  Composition shape: {sample['composition'].shape}")
        print(f"  Composition values (first 5): {sample['composition'][:5]}")
    
    # Test bariatric dataset
    print("\n2. Testing Bariatric Dataset:")
    ds_bariatric = BariatricDataset.from_directory(
        DATA_ROOT,
        mode='train',
        use_all_scans=True,
        n_vertices=10000,
        augment=False
    )
    
    # Test sarcopenia dataset
    print("\n3. Testing Sarcopenia Dataset:")
    ds_sarcopenia = SarcopeniaDataset.from_directory(
        DATA_ROOT,
        mode='train',
        n_vertices=10000,
        augment=False
    )
    
    # Test combined dataset
    print("\n4. Testing Combined Dataset:")
    ds_combined = create_combined_dataset(
        DATA_ROOT,
        mode='train',
        populations=['normal', 'bariatric', 'sarcopenia'],
        n_vertices=10000,
        augment=False
    )
    
    print(f"\n Total combined samples: {len(ds_combined)}")