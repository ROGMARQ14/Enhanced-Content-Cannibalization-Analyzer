"""
Robust data loading utilities for handling various CSV formats and encodings.
"""

import pandas as pd
import chardet
import io
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles robust CSV loading with encoding and delimiter detection."""
    
    @staticmethod
    def detect_encoding(file_content: bytes) -> str:
        """Detect file encoding using chardet."""
        result = chardet.detect(file_content)
        return result['encoding'] or 'utf-8'
    
    @staticmethod
    def detect_delimiter(sample: str) -> str:
        """Detect CSV delimiter from sample content."""
        delimiters = [',', ';', '\t', '|']
        lines = sample.split('\n')[:5]  # Check first 5 lines
        
        for delimiter in delimiters:
            if all(delimiter in line for line in lines if line.strip()):
                return delimiter
        
        return ','  # Default to comma
    
    @classmethod
    def load_csv(cls, file, file_name: str = "uploaded_file") -> pd.DataFrame:
        """
        Robust CSV loader that handles encoding and delimiter issues.
        
        Args:
            file: File-like object or file path
            file_name: Name for error reporting
            
        Returns:
            pandas.DataFrame
        """
        try:
            # Read raw content
            if hasattr(file, 'read'):
                file_content = file.read()
                file.seek(0)  # Reset file pointer
            else:
                with open(file, 'rb') as f:
                    file_content = f.read()
            
            # Detect encoding
            encoding = cls.detect_encoding(file_content)
            logger.info(f"Detected encoding: {encoding} for {file_name}")
            
            # Decode content
            try:
                content_str = file_content.decode(encoding)
            except UnicodeDecodeError:
                # Fallback to utf-8 with error handling
                content_str = file_content.decode('utf-8', errors='ignore')
                logger.warning(f"Used fallback decoding for {file_name}")
            
            # Detect delimiter
            delimiter = cls.detect_delimiter(content_str)
            logger.info(f"Detected delimiter: '{delimiter}' for {file_name}")
            
            # Load DataFrame
            if hasattr(file, 'read'):
                file.seek(0)
                df = pd.read_csv(file, sep=delimiter, encoding=encoding, on_bad_lines='skip')
            else:
                df = pd.read_csv(file, sep=delimiter, encoding=encoding, on_bad_lines='skip')
            
            logger.info(f"Successfully loaded {len(df)} rows from {file_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {file_name}: {str(e)}")
            raise ValueError(f"Could not parse {file_name}: {str(e)}")
    
    @staticmethod
    def validate_seo_data(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate SEO data structure and identify key columns."""
        
        validation_result = {
            'valid': False,
            'url_column': None,
            'title_column': None,
            'h1_column': None,
            'meta_column': None,
            'embedding_column': None,
            'issues': []
        }
        
        if df.empty:
            validation_result['issues'].append("Empty dataset")
            return validation_result
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Find URL column
        url_candidates = ['URL', 'Address', 'url', 'address', 'Page', 'page']
        for candidate in url_candidates:
            matches = [col for col in df.columns if candidate.lower() in col.lower()]
            if matches:
                validation_result['url_column'] = matches[0]
                break
        
        if not validation_result['url_column']:
            validation_result['issues'].append("No URL column found")
            return validation_result
        
        # Find other columns
        column_mappings = {
            'title_column': ['Title 1', 'Title', 'title', 'Page Title', 'Meta Title'],
            'h1_column': ['H1-1', 'H1', 'h1', 'H1 Tag', 'Header 1'],
            'meta_column': ['Meta Description 1', 'Meta Description', 'meta description', 'Description'],
            'embedding_column': ['embedding', 'Embedding', 'Vector', 'vector']
        }
        
        for key, candidates in column_mappings.items():
            for candidate in candidates:
                matches = [col for col in df.columns if candidate.lower() in col.lower()]
                if matches:
                    validation_result[key] = matches[0]
                    break
        
        # Check for required columns
        if data_type == 'internal':
            if not validation_result['title_column']:
                validation_result['issues'].append("No title column found")
            if not validation_result['h1_column']:
                validation_result['issues'].append("No H1 column found")
        
        validation_result['valid'] = len(validation_result['issues']) == 0
        return validation_result
