"""
Robust data loading utilities for CSV files with comprehensive error handling.
"""

import pandas as pd
import chardet
import logging
from typing import Dict, Any, List
import io
import csv

logger = logging.getLogger(__name__)


class DataLoader:
    """Robust CSV data loader with encoding and delimiter detection."""
    
    @staticmethod
    def detect_encoding(file_content: bytes) -> str:
        """Detect file encoding from content."""
        try:
            result = chardet.detect(file_content)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence'] or 0.0
            
            logger.info(
                f"Detected encoding: {encoding} "
                f"(confidence: {confidence:.2f})"
            )
            return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return 'utf-8'
    
    @staticmethod
    def detect_delimiter(sample: str) -> str:
        """Detect CSV delimiter from sample."""
        try:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            logger.info(f"Detected delimiter: '{delimiter}'")
            return delimiter
        except Exception:
            logger.warning("Delimiter detection failed, using comma")
            return ','
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """Clean and standardize column names."""
        cleaned = []
        for col in columns:
            if pd.isna(col):
                cleaned.append('Unnamed_Column')
            else:
                col_str = str(col).strip()
                col_str = col_str.replace('\n', ' ').replace('\r', ' ')
                col_str = col_str.replace('"', '').replace("'", '')
                col_str = col_str.lower()
                cleaned.append(col_str)
        return cleaned
    
    @staticmethod
    def load_csv(file, file_type: str = "data") -> pd.DataFrame:
        """Load CSV file with robust error handling."""
        logger.info(f"Loading {file_type} file: {file.name}")
        
        try:
            # Read file content
            file_content = file.read()
            
            # Detect encoding
            encoding = DataLoader.detect_encoding(file_content)
            
            # Decode content
            try:
                content_str = file_content.decode(encoding)
            except UnicodeDecodeError:
                # Fallback encodings
                fallback_encodings = ['latin1', 'cp1252', 'iso-8859-1']
                for fallback_encoding in fallback_encodings:
                    try:
                        content_str = file_content.decode(fallback_encoding)
                        logger.info(
                            f"Used fallback encoding: {fallback_encoding}"
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(
                        "Could not decode file with any encoding"
                    )
            
            # Detect delimiter
            delimiter = DataLoader.detect_delimiter(content_str[:2000])
            
            # Read CSV
            na_values = ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN']
            df = pd.read_csv(
                io.StringIO(content_str),
                delimiter=delimiter,
                on_bad_lines='warn',
                skip_blank_lines=True,
                na_values=na_values,
                keep_default_na=True,
                low_memory=False,
                dtype=str
            )
            
            # Clean column names
            df.columns = DataLoader.clean_column_names(df.columns)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            logger.info(
                f"Successfully loaded {len(df)} rows "
                f"and {len(df.columns)} columns"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise ValueError(f"Failed to load {file_type}: {str(e)}")
    
    @staticmethod
    def validate_seo_data(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate SEO data structure."""
        validation = {
            'valid': True,
            'issues': [],
            'url_column': None,
            'title_column': None,
            'h1_column': None,
            'meta_column': None
        }
        
        if df.empty:
            validation['valid'] = False
            validation['issues'].append("DataFrame is empty")
            return validation
        
        columns = df.columns.tolist()
        
        if data_type == 'internal':
            # Find URL column
            url_candidates = ['url', 'address', 'page', 'uri', 'link']
            for col in columns:
                if any(candidate in col for candidate in url_candidates):
                    validation['url_column'] = col
                    break
            
            if not validation['url_column']:
                validation['valid'] = False
                validation['issues'].append("No URL column found")
            
            # Find Title column
            title_candidates = ['title', 'title1', 'page title', 'meta title']
            for col in columns:
                if any(candidate in col for candidate in title_candidates):
                    validation['title_column'] = col
                    break
            
            if not validation['title_column']:
                validation['valid'] = False
                validation['issues'].append("No Title column found")
            
            # Find H1 column
            h1_candidates = ['h1', 'h1-1', 'heading1', 'header1']
            for col in columns:
                if any(candidate in col for candidate in h1_candidates):
                    validation['h1_column'] = col
                    break
            
            if not validation['h1_column']:
                validation['valid'] = False
                validation['issues'].append("No H1 column found")
            
            # Find Meta Description column (optional)
            meta_candidates = [
                'meta description', 'meta desc', 'description',
                'meta_description'
            ]
            for col in columns:
                if any(candidate in col for candidate in meta_candidates):
                    validation['meta_column'] = col
                    break
        
        elif data_type == 'gsc':
            # Find URL/Page column
            url_candidates = ['url', 'page', 'landing page', 'landing_page']
            for col in columns:
                if any(candidate in col for candidate in url_candidates):
                    validation['url_column'] = col
                    break
            
            if not validation['url_column']:
                validation['valid'] = False
                validation['issues'].append("No URL/Page column found")
            
            # Find Query column
            query_candidates = [
                'query', 'keyword', 'search term', 'search_term'
            ]
            for col in columns:
                if any(candidate in col for candidate in query_candidates):
                    validation['title_column'] = col
                    break
            
            if not validation['title_column']:
                validation['valid'] = False
                validation['issues'].append("No Query column found")
        
        return validation
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the loaded data."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict('records')
        }
