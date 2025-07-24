#!/usr/bin/env python3
"""
Test script to verify CSV parsing fixes for the Enhanced Content Cannibalization Analyzer.

This script tests the robust CSV parsing capabilities with various edge cases
and provides clear feedback on what's working and what needs attention.
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
from io import StringIO

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_loader import DataLoader

def create_test_files():
    """Create test CSV files with various formats and edge cases."""
    
    test_files = {}
    
    # Test 1: Standard internal SEO report
    internal_standard = """Address,Title 1,H1-1,Meta Description 1
https://example.com/page1,Best SEO Tools 2024,Best SEO Tools,Discover the top SEO tools for 2024
https://example.com/page2,SEO Tool Reviews,SEO Tool Reviews,Comprehensive reviews of SEO tools
https://example.com/page3,Free SEO Tools,Free SEO Tools,List of free SEO tools for beginners"""
    
    test_files['internal_standard.csv'] = internal_standard
    
    # Test 2: Internal report with different column names
    internal_varied = """URL,Page Title,Main Heading,Description
https://example.com/page1,Best SEO Tools 2024,Best SEO Tools,Discover the top SEO tools for 2024
https://example.com/page2,SEO Tool Reviews,SEO Tool Reviews,Comprehensive reviews of SEO tools"""
    
    test_files['internal_varied.csv'] = internal_varied
    
    # Test 3: GSC standard format
    gsc_standard = """Landing page,Query
https://example.com/page1,best seo tools
https://example.com/page1,seo tools 2024
https://example.com/page2,seo tool reviews
https://example.com/page3,free seo tools"""
    
    test_files['gsc_standard.csv'] = gsc_standard
    
    # Test 4: GSC with different column names
    gsc_varied = """Page,Search Term
https://example.com/page1,best seo tools
https://example.com/page1,seo tools 2024
https://example.com/page2,seo tool reviews"""
    
    test_files['gsc_varied.csv'] = gsc_varied
    
    # Test 5: Files with encoding issues (UTF-8 BOM)
    internal_utf8_bom = "\ufeffAddress,Title 1,H1-1,Meta Description 1\nhttps://example.com/page1,Título con acentos,H1 con acentos,Descripción con acentos"
    
    test_files['internal_utf8_bom.csv'] = internal_utf8_bom
    
    # Test 6: Files with different delimiters
    internal_semicolon = """Address;Title 1;H1-1;Meta Description 1
https://example.com/page1;Best SEO Tools 2024;Best SEO Tools;Discover the top SEO tools"""
    
    test_files['internal_semicolon.csv'] = internal_semicolon
    
    # Test 7: Files with missing data
    internal_missing = """Address,Title 1,H1-1,Meta Description 1
https://example.com/page1,Best SEO Tools 2024,Best SEO Tools,
https://example.com/page2,,SEO Tool Reviews,Comprehensive reviews"""
    
    test_files['internal_missing.csv'] = internal_missing
    
    return test_files

def test_data_loader():
    """Test the DataLoader with various file formats."""
    
    print("🧪 Testing CSV Parsing Fixes...")
    print("=" * 50)
    
    test_files = create_test_files()
    results = []
    
    for filename, content in test_files.items():
        print(f"\n📁 Testing: {filename}")
        print("-" * 30)
        
        temp_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            # Test loading
            with open(temp_path, 'rb') as f:
                # Create mock file object with name attribute
                class MockFile:
                    def __init__(self, file_obj, name):
                        self.file = file_obj
                        self.name = name
                    
                    def read(self):
                        return self.file.read()
                
                mock_file = MockFile(open(temp_path, 'rb'), filename)
                df = DataLoader.load_csv(mock_file, filename)
                
                print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                print(f"   Columns: {list(df.columns)}")
                
                # Test validation
                file_type = 'internal' if 'internal' in filename else 'gsc'
                validation = DataLoader.validate_seo_data(df, file_type)
                
                if validation['valid']:
                    print(f"✅ Validation passed")
                    print(f"   URL column: {validation['url_column']}")
                    print(f"   Title/Query column: {validation['title_column']}")
                    if file_type == 'internal':
                        print(f"   H1 column: {validation['h1_column']}")
                else:
                    print(f"❌ Validation failed: {validation['issues']}")
                    if 'suggestions' in validation:
                        print(f"   Available columns: {validation['suggestions']['available_columns']}")
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'validation': validation
                })
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (PermissionError, OSError):
                    # Windows file locking issue - ignore
                    pass
    
    return results

def test_edge_cases():
    """Test edge cases that commonly cause parsing errors."""
    
    print("\n🎯 Testing Edge Cases...")
    print("=" * 50)
    
    edge_cases = [
        ("Empty file", ""),
        ("Single row", "Address,Title 1,H1-1,Meta Description 1\nhttps://test.com,Test,Test H1,Test desc"),
        ("Special characters", "Address,Title 1,H1-1,Meta Description 1\nhttps://test.com,Test with 'quotes' and \"double quotes\",Test H1,Test with commas, and semicolons;"),
        ("Unicode content", "Address,Title 1,H1-1,Meta Description 1\nhttps://test.com,Título español éñ,H1 con ümläüts,Descripción française"),
        ("Very long content", "Address,Title 1,H1-1,Meta Description 1\n" + "https://test.com," + "Very long title " * 50 + ",Very long H1 " * 50 + ",Very long description " * 100),
    ]
    
    for case_name, content in edge_cases:
        print(f"\n🧪 Testing: {case_name}")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            with open(temp_path, 'rb') as f:
                class MockFile:
                    def __init__(self, file_obj, name):
                        self.file = file_obj
                        self.name = name
                    
                    def read(self):
                        return self.file.read()
                
                mock_file = MockFile(open(temp_path, 'rb'), f"edge_{case_name}.csv")
                df = DataLoader.load_csv(mock_file, f"edge_{case_name}.csv")
                
                print(f"✅ Success: {len(df)} rows loaded")
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
        finally:
            if 'temp_path' in locals():
                os.unlink(temp_path)

def generate_report(results):
    """Generate a summary report of the test results."""
    
    print("\n📊 Test Summary Report")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ Failed tests:")
        for result in results:
            if not result['success']:
                print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
    
    print("\n✅ All tests completed!")
    return passed_tests == total_tests

def main():
    """Main test function."""
    
    print("🚀 Enhanced Content Cannibalization Analyzer - CSV Fix Tests")
    print("=" * 60)
    
    # Test basic functionality
    results = test_data_loader()
    
    # Test edge cases
    test_edge_cases()
    
    # Generate final report
    all_passed = generate_report(results)
    
    if all_passed:
        print("\n🎉 All tests passed! The CSV parsing fixes are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
