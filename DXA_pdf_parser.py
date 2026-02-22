import PyPDF2
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings


class DXACompositionParser:
    """
    Parser for (DXA) Total Body Composition Ancillary.pdf"
    """
    
    # Unit conversions
    LBS_TO_KG = 0.453592
    LBS_TO_GRAMS = 453.592
    
    # Measurement regions
    REGIONS = [
        'Arms', 'Arm Right', 'Arm Left', 'Arms Diff.',
        'Legs', 'Leg Right', 'Leg Left', 'Legs Diff.',
        'Trunk', 'Trunk Right', 'Trunk Left', 'Trunk Diff.',
        'Android', 'Gynoid',
        'Total', 'Total Right', 'Total Left', 'Total Diff.'
    ]
    
    def __init__(self, convert_to_metric: bool = True):
        """ 
        Args:
            convert_to_metric: If True, convert lbs to kg/grams for compatibility
        """
        self.convert_to_metric = convert_to_metric
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {pdf_path}: {e}")
    
    def parse_regional_measurements(self, text: str) -> Dict[str, float]:
        """
        Expected format:
        Region  Tissue(%Fat)  Region(%Fat)  Tissue(lbs)  Fat(lbs)  Lean(lbs)  BMC(lbs)  TotalMass(lbs)
        """
        measurements = {}
        
        for region in self.REGIONS:
            # Create pattern to match the row
            # Region name followed by 7 numbers (some may be negative)
            pattern = rf'{re.escape(region)}\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)\s+([-]?\d+\.?\d*)'
            
            match = re.search(pattern, text)
            if match:
                # Clean region name for dict key
                region_key = region.lower().replace(' ', '_').replace('.', '')
                
                # Extract values
                tissue_fat_pct = float(match.group(1))
                region_fat_pct = float(match.group(2))
                tissue_mass = float(match.group(3))
                fat_mass = float(match.group(4))
                lean_mass = float(match.group(5))
                bmc = float(match.group(6))
                total_mass = float(match.group(7))
                
                # Convert to metric if requested
                if self.convert_to_metric:
                    tissue_mass_kg = tissue_mass * self.LBS_TO_KG
                    fat_mass_kg = fat_mass * self.LBS_TO_KG
                    lean_mass_kg = lean_mass * self.LBS_TO_KG
                    bmc_grams = bmc * self.LBS_TO_GRAMS
                    total_mass_kg = total_mass * self.LBS_TO_KG
                else:
                    tissue_mass_kg = tissue_mass
                    fat_mass_kg = fat_mass
                    lean_mass_kg = lean_mass
                    bmc_grams = bmc
                    total_mass_kg = total_mass
                
                # Store measurements
                measurements[f'{region_key}_tissue_fat_pct'] = tissue_fat_pct
                measurements[f'{region_key}_region_fat_pct'] = region_fat_pct
                measurements[f'{region_key}_tissue_mass'] = tissue_mass_kg
                measurements[f'{region_key}_fat_mass'] = fat_mass_kg
                measurements[f'{region_key}_lean_mass'] = lean_mass_kg
                measurements[f'{region_key}_bmc'] = bmc_grams
                measurements[f'{region_key}_total_mass'] = total_mass_kg
        
        return measurements
    
    def parse_derived_metrics(self, text: str) -> Dict[str, float]:
        """Parse derived metrics (RMR, RSMI, VAT, SAT)"""
        metrics = {}
        
        # Resting Metabolic Rate (RMR)
        rmr_match = re.search(r'(\d+,?\d*)\s+cal/day', text)
        if rmr_match:
            rmr_str = rmr_match.group(1).replace(',', '')
            metrics['rmr'] = float(rmr_str)
        
        # Relative Skeletal Muscle Index (RSMI)
        rsmi_match = re.search(r'([\d.]+)\s+kg/m', text)
        if rsmi_match:
            metrics['rsmi'] = float(rsmi_match.group(1))
        
        # Visceral Adipose Tissue (VAT) - may not be in all reports
        vat_volume_match = re.search(r'([\d.]+)\s+in³.*?Volume', text, re.DOTALL)
        if vat_volume_match:
            metrics['vat_volume'] = float(vat_volume_match.group(1))
        
        vat_mass_match = re.search(r'([\d.]+)\s+lbs.*?Mass.*?Visceral', text, re.DOTALL)
        if vat_mass_match:
            vat_mass_lbs = float(vat_mass_match.group(1))
            if self.convert_to_metric:
                metrics['vat_mass'] = vat_mass_lbs * self.LBS_TO_GRAMS  # grams
            else:
                metrics['vat_mass'] = vat_mass_lbs
        
        vat_area_match = re.search(r'([\d.]+)\s+in².*?Area.*?Visceral', text, re.DOTALL)
        if vat_area_match:
            metrics['vat_area'] = float(vat_area_match.group(1))
        
        # Subcutaneous Adipose Tissue (SAT)
        sat_volume_match = re.search(r'([\d.]+)\s+in³.*?Volume.*?Subcutaneous', text, re.DOTALL)
        if sat_volume_match:
            metrics['sat_volume'] = float(sat_volume_match.group(1))
        
        sat_mass_match = re.search(r'([\d.]+)\s+lbs.*?Mass.*?Subcutaneous', text, re.DOTALL)
        if sat_mass_match:
            sat_mass_lbs = float(sat_mass_match.group(1))
            if self.convert_to_metric:
                metrics['sat_mass'] = sat_mass_lbs * self.LBS_TO_GRAMS
            else:
                metrics['sat_mass'] = sat_mass_lbs
        
        sat_area_match = re.search(r'([\d.]+)\s+in².*?Area.*?Subcutaneous', text, re.DOTALL)
        if sat_area_match:
            metrics['sat_area'] = float(sat_area_match.group(1))
        
        # Fat Mass Ratios (sarcopenia dataset)
        trunk_ratio_match = re.search(r'Trunk Fat Mass/Total Fat Mass\s+([\d.]+)', text)
        if trunk_ratio_match:
            metrics['trunk_fat_ratio'] = float(trunk_ratio_match.group(1))
        
        legs_ratio_match = re.search(r'Legs Fat Mass/Total Fat Mass\s+([\d.]+)', text)
        if legs_ratio_match:
            metrics['legs_fat_ratio'] = float(legs_ratio_match.group(1))
        
        limbs_trunk_match = re.search(r'Limbs Fat Mass/Trunk Fat Mass\s+([\d.]+)', text)
        if limbs_trunk_match:
            metrics['limbs_trunk_ratio'] = float(limbs_trunk_match.group(1))
        
        return metrics
    
    def parse_metadata(self, text: str) -> Dict[str, any]:
        """Parse demographic info"""
        metadata = {}
        
        # Patient ID
        patient_id_match = re.search(r'Patient ID:\s*(\S+)', text)
        if patient_id_match:
            metadata['patient_id'] = patient_id_match.group(1)
        
        # Age
        age_match = re.search(r'Age:\s*([\d.]+)\s*years', text)
        if age_match:
            metadata['age'] = float(age_match.group(1))
        
        # Height
        height_match = re.search(r'Height:\s*([\d.]+)\s*in', text)
        if height_match:
            metadata['height_inches'] = float(height_match.group(1))
            metadata['height_cm'] = float(height_match.group(1)) * 2.54
        
        # Weight
        weight_match = re.search(r'Weight:\s*([\d.]+)\s*lbs', text)
        if weight_match:
            metadata['weight_lbs'] = float(weight_match.group(1))
            metadata['weight_kg'] = float(weight_match.group(1)) * self.LBS_TO_KG
        
        # Sex
        sex_match = re.search(r'Sex:\s*(\w+)', text)
        if sex_match:
            metadata['sex'] = sex_match.group(1)
        
        # Ethnicity
        ethnicity_match = re.search(r'Ethnicity:\s*([^\n]+)', text)
        if ethnicity_match:
            metadata['ethnicity'] = ethnicity_match.group(1).strip()
        
        # Scan date
        measured_match = re.search(r'Measured:\s*([^\(]+)', text)
        if measured_match:
            metadata['scan_date'] = measured_match.group(1).strip()
        
        # Scanner model
        if 'Lunar iDXA' in text:
            metadata['scanner_model'] = 'GE Lunar iDXA'
        
        return metadata
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, float]:
        """
        Parse a single PDF
        Args:
            pdf_path: Path to PDF
            
        Returns:
            measurements: Dictionary with all extracted values
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Parse different sections
        measurements = {}
        
        # Regional measurements
        regional = self.parse_regional_measurements(text)
        measurements.update(regional)
        
        # Derived metrics
        derived = self.parse_derived_metrics(text)
        measurements.update(derived)
        
        # Metadata
        metadata = self.parse_metadata(text)
        measurements.update(metadata)
        
        # Add source info
        measurements['pdf_path'] = str(pdf_path)
        
        return measurements
    
    def parse_directory(self, directory: str, pattern: str = "**/composition.pdf") -> pd.DataFrame:
        """
        Parse all PDFs in a directory
        Args:
            directory: Root directory
            pattern: Glob pattern
        Returns:
            DataFrame with all measurements
        """
        pdf_files = list(Path(directory).glob(pattern))
        
        if len(pdf_files) == 0:
            warnings.warn(f"No PDFs found matching pattern '{pattern}' in {directory}")
            return pd.DataFrame()
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        failed = []
        
        for pdf_file in pdf_files:
            try:
                measurements = self.parse_pdf(str(pdf_file))
                
                # Add subject ID from path
                # Extract from folder name (e.g., "1_1" from "dxa/ba/1_1/composition.pdf")
                subject_id = pdf_file.parent.name
                measurements['subject_id'] = subject_id
                
                results.append(measurements)
            except Exception as e:
                print(f"Failed to parse {pdf_file}: {e}")
                failed.append(str(pdf_file))
        
        if failed:
            print(f"\nFailed to parse {len(failed)} PDFs:")
            for f in failed[:5]:
                print(f"  - {f}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        df = pd.DataFrame(results)
        print(f"\nSuccessfully parsed {len(df)} PDFs")
        print(f"Extracted {len(df.columns)} fields per PDF")
        
        return df
    
    def validate_measurements(self, measurements: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate extracted measurements
        Returns:
            is_valid: Whether all measurements pass validation
            warnings
        """
        validation_warnings = []
        
        # Define reasonable ranges (in kg or grams after conversion)
        if self.convert_to_metric:
            ranges = {
                'total_fat_mass': (1, 100),  # kg
                'total_lean_mass': (20, 120),  # kg
                'total_bmc': (500, 5000),  # grams
                'total_total_mass': (30, 200),  # kg
                'vat_mass': (50, 10000),  # grams
                'vat_volume': (50, 5000),  # cm³
                'android_fat_mass': (0.5, 30),  # kg
                'gynoid_fat_mass': (1, 40),  # kg
                'rmr': (800, 3000),  # cal/day
                'rsmi': (3, 15),  # kg/m²
            }
        else:
            ranges = {
                'total_fat_mass': (2, 220),  # lbs
                'total_lean_mass': (44, 265),  # lbs
                'total_bmc': (1, 11),  # lbs
            }
        
        for measurement, (min_val, max_val) in ranges.items():
            if measurement in measurements:
                value = measurements[measurement]
                if not (min_val <= value <= max_val):
                    warnings.append(
                        f"{measurement} = {value:.2f} outside expected range [{min_val}, {max_val}]"
                    )
        
        is_valid = len(validation_warnings) == 0
        return is_valid, validation_warnings


def parse_all_datasets(data_root: str = "D:/human") -> Dict[str, pd.DataFrame]:
    """
    Parse all datasets
    Args:
        data_root: Root data directory
    Returns:
        Dictionary with subsets DataFrames
    """
    parser = DXACompositionParser(convert_to_metric=True)
    
    data_root = Path(data_root)
    
    results = {}
    
    # Normal dataset
    print("\n" + "="*80)
    print("Parsing NORMAL dataset")
    print("="*80)
    normal_dir = data_root / "dxa" / "ba"
    df_normal = parser.parse_directory(str(normal_dir), pattern="*/composition.pdf")
    results['normal'] = df_normal
    
    # Bariatric dataset
    print("\n" + "="*80)
    print("Parsing BARIATRIC dataset")
    print("="*80)
    bariatric_dir = data_root / "dxa" / "ba"
    df_bariatric = parser.parse_directory(str(bariatric_dir), pattern="*/composition.pdf")
    results['bariatric'] = df_bariatric
    
    # Sarcopenia dataset
    print("\n" + "="*80)
    print("Parsing SARCOPENIA dataset")
    print("="*80)
    sarcopenia_dir = data_root / "scans" / "Sarcopenia" / "OIM study"
    df_sarcopenia = parser.parse_directory(str(sarcopenia_dir), pattern="*/*.pdf")
    results['sarcopenia'] = df_sarcopenia
    
    return results


if __name__ == "__main__":
    import sys
    
    print("DXA Composition PDF Parser")
    print("="*80)
    
    # Test on example file
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "D:/human/dxa/ba/1_1/composition.pdf"
    
    print(f"\nTesting parser on: {pdf_path}")
    
    parser = DXACompositionParser(convert_to_metric=True)
    
    if Path(pdf_path).exists():
        measurements = parser.parse_pdf(pdf_path)
        
        print(f"\nExtracted {len(measurements)} measurements:")
        print("-"*80)
        
        # Group by category
        regional = {k: v for k, v in measurements.items() if any(r.lower().replace(' ', '_').replace('.', '') in k for r in parser.REGIONS)}
        derived = {k: v for k, v in measurements.items() if k in ['rmr', 'rsmi', 'vat_mass', 'vat_volume', 'sat_mass', 'sat_volume']}
        metadata = {k: v for k, v in measurements.items() if k in ['patient_id', 'age', 'height_cm', 'weight_kg', 'sex', 'ethnicity']}
        
        print("\nMetadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        
        print("\nDerived Metrics:")
        for k, v in derived.items():
            print(f"  {k}: {v}")
        
        print(f"\nRegional Measurements: {len(regional)} values")
        print("  (Sample):")
        for i, (k, v) in enumerate(list(regional.items())[:10]):
            print(f"  {k}: {v:.2f}")
        
        # Validate
        is_valid, warnings = parser.validate_measurements(measurements)
        if is_valid:
            print("\n✓ All measurements within expected ranges")
        else:
            print("\n⚠ Validation warnings:")
            for w in warnings:
                print(f"  - {w}")
    else:
        print(f"File not found: {pdf_path}")
        print("\nUsage: python pdf_parser.py [path/to/composition.pdf]")
