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
        """Validate SEO data structure with enhanced column detection."""
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
        
        # Log available columns for debugging
        logger.info(f"Available columns for {data_type}: {columns}")
        
        if data_type == 'internal':
            # Enhanced URL column detection
            url_patterns = [
                'url', 'address', 'page', 'uri', 'link', 'path', 'location',
                'webpage', 'site', 'href', 'canonical', 'permalink'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in url_patterns):
                    validation['url_column'] = col
                    break
            
            if not validation['url_column']:
                # Try exact match first
                for col in columns:
                    if col.lower() in ['url', 'address', 'page']:
                        validation['url_column'] = col
                        break
            
            if not validation['url_column']:
                validation['valid'] = False
                validation['issues'].append("No URL column found")
            
            # Enhanced Title column detection
            title_patterns = [
                'title', 'page title', 'meta title', 'seo title', 'browser title',
                'titulo', 'titre', 'titel', 'og:title', 'twitter:title'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in title_patterns):
                    validation['title_column'] = col
                    break
            
            if not validation['title_column']:
                # Try exact match
                for col in columns:
                    if col.lower() in ['title', 'title1', 'title 1']:
                        validation['title_column'] = col
                        break
            
            if not validation['title_column']:
                validation['valid'] = False
                validation['issues'].append("No Title column found")
            
            # Enhanced H1 column detection
            h1_patterns = [
                'h1', 'heading1', 'header1', 'main heading', 'primary heading',
                'h1 tag', 'h1-1', 'h1_1', 'heading 1'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in h1_patterns):
                    validation['h1_column'] = col
                    break
            
            if not validation['h1_column']:
                # Try exact match
                for col in columns:
                    if col.lower() in ['h1', 'h1-1', 'h1_1']:
                        validation['h1_column'] = col
                        break
            
            if not validation['h1_column']:
                validation['valid'] = False
                validation['issues'].append("No H1 column found")
            
            # Enhanced Meta Description column detection (optional)
            meta_patterns = [
                'meta description', 'meta desc', 'description', 'meta_description',
                'meta-desc', 'meta:description', 'og:description', 'twitter:description',
                'seo description', 'snippet', 'summary'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in meta_patterns):
                    validation['meta_column'] = col
                    break
        
        elif data_type == 'gsc':
            # Enhanced URL/Page column detection for GSC
            url_patterns = [
                'url', 'page', 'landing page', 'landing_page', 'landingpage',
                'webpage', 'site', 'path', 'location', 'uri', 'link', 'href',
                'canonical', 'permalink', 'address'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in url_patterns):
                    validation['url_column'] = col
                    break
            
            if not validation['url_column']:
                # Try exact match
                for col in columns:
                    if col.lower() in ['url', 'page', 'landing page']:
                        validation['url_column'] = col
                        break
            
            if not validation['url_column']:
                validation['valid'] = False
                validation['issues'].append("No URL/Page column found")
            
            # Enhanced Query column detection for GSC
            query_patterns = [
                'query', 'keyword', 'search term', 'search_term', 'searchterm',
                'search query', 'searchquery', 'key word', 'keyterm', 'term',
                'search phrase', 'phrase', 'search', 'keywords'
            ]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in query_patterns):
                    validation['title_column'] = col
                    break
            
            if not validation['title_column']:
                # Try exact match
                for col in columns:
                    if col.lower() in ['query', 'keyword', 'search term']:
                        validation['title_column'] = col
                        break
            
            if not validation['title_column']:
                validation['valid'] = False
                validation['issues'].append("No Query column found")
        
        # Provide helpful suggestions
        if not validation['valid']:
            validation['suggestions'] = {
                'available_columns': columns,
                'expected_patterns': {
                    'internal': {
                        'url': ['url', 'address', 'page', 'uri', 'link'],
                        'title': ['title', 'page title', 'meta title'],
                        'h1': ['h1', 'heading1', 'header1'],
                        'meta': ['meta description', 'description', 'meta desc']
                    },
                    'gsc': {
                        'url': ['url', 'page', 'landing page', 'landing_page'],
                        'query': ['query', 'keyword', 'search term', 'search_term']
                    }
                }
            }
        
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
