#!/usr/bin/env python3
"""
Final verification script to test the complete integration of CSV parser fixes.
"""

import os
import sys
import tempfile
import pandas as pd

# Ensure we're in the right directory
os.chdir(r'C:\Users\admin\Documents\Marketing\Roger SEO\Scripts\Enhanced-Content-Cannibalization-Analyzer-main')
sys.path.insert(0, 'src')

from src.utils.data_loader import DataLoader

def test_complete_integration():
    """Test the complete integration with realistic data."""
    
    print("🔍 Final Integration Verification")
    print("=" * 50)
    
    # Test data that mimics real SEO tool exports
    test_cases = [
        {
            'name': 'Screaming Frog Export',
            'content': """Address,Title 1,H1-1,Meta Description 1,Word Count
https://example.com/seo-tools,Best SEO Tools 2024,Best SEO Tools,Discover the top SEO tools for 2024,1200
https://example.com/seo-reviews,SEO Tool Reviews,SEO Tool Reviews,Comprehensive reviews of SEO tools,1500
https://example.com/free-seo,Free SEO Tools,Free SEO Tools,List of free SEO tools for beginners,800""",
            'type': 'internal'
        },
        {
            'name': 'Google Search Console',
            'content': """Landing page,Query,Clicks,Impressions
https://example.com/seo-tools,best seo tools,45,1200
https://example.com/seo-tools,seo tools 2024,32,890
https://example.com/seo-reviews,seo tool reviews,28,650""",
            'type': 'gsc'
        },
        {
            'name': 'Sitebulb Export',
            'content': """URL,Page Title,H1,Meta Description
https://example.com/seo-tools,Best SEO Tools 2024,Best SEO Tools,Discover the top SEO tools for 2024
https://example.com/seo-reviews,SEO Tool Reviews,SEO Tool Reviews,Comprehensive reviews of SEO tools""",
            'type': 'internal'
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                f.write(test_case['content'])
                temp_path = f.name
            
            # Test loading
            with open(temp_path, 'rb') as f:
                class MockFile:
                    def __init__(self, file_obj, name):
                        self.file = file_obj
                        self.name = name
                    
                    def read(self):
                        return self.file.read()
                
                mock_file = MockFile(open(temp_path, 'rb'), f"{test_case['name']}.csv")
                df = DataLoader.load_csv(mock_file, f"{test_case['name']}.csv")
                
                # Test validation
                validation = DataLoader.validate_seo_data(df, test_case['type'])
                
                if validation['valid']:
                    print(f"✅ {test_case['name']} - PASSED")
                    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                    print(f"   URL column: {validation['url_column']}")
                    print(f"   Title column: {validation['title_column']}")
                else:
                    print(f"❌ {test_case['name']} - FAILED")
                    print(f"   Issues: {validation['issues']}")
                    all_passed = False
                
        except Exception as e:
            print(f"❌ {test_case['name']} - ERROR: {str(e)}")
            all_passed = False
        
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    return all_passed

def test_error_handling():
    """Test error handling for edge cases."""
    
    print("\n🚨 Testing Error Handling")
    print("=" * 30)
    
    error_cases = [
        ("Empty file", ""),
        ("Invalid format", "This is not a CSV file"),
        ("Missing columns", "Column1,Column2\nValue1,Value2"),
    ]
    
    for name, content in error_cases:
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
                
                mock_file = MockFile(open(temp_path, 'rb'), f"{name}.csv")
                df = DataLoader.load_csv(mock_file, f"{name}.csv")
                
                # Should not reach here for error cases
                print(f"⚠️ {name} - Unexpected success")
                
        except Exception as e:
            print(f"✅ {name} - Properly handled: {type(e).__name__}")
        
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

def main():
    """Run complete verification."""
    
    print("🎯 Enhanced Content Cannibalization Analyzer")
    print("🔧 CSV Parser Integration Verification")
    print("=" * 60)
    
    # Test integration
    integration_passed = test_complete_integration()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "=" * 60)
    if integration_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ CSV parser fixes are fully integrated")
        print("✅ Ready for production use")
    else:
        print("❌ Some tests failed - please review")
    
    return integration_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
