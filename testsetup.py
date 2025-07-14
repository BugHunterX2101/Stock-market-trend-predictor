#!/usr/bin/env python3
"""
Test script to verify the stock prediction setup
"""

import sys
import importlib

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'yfinance',
        'plotly',
        'sklearn',
        'tensorflow'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"{package}: {version}")
        except ImportError as e:
            print(f"{package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_file_syntax():
    """Test Python file syntax"""
    print("\nTesting file syntax...")
    
    files_to_test = ['app.py', 'create_sample_model.py']
    
    for filename in files_to_test:
        try:
            with open(filename, 'r') as f:
                compile(f.read(), filename, 'exec')
            print(f" {filename}: Syntax OK")
        except SyntaxError as e:
            print(f" {filename}: Syntax Error - {e}")
            return False
        except FileNotFoundError:
            print(f" {filename}: File not found")
            return False
    
    return True

def main():
    print("Stock Prediction App Setup Test")
    print("=" * 40)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test file syntax
    syntax_ok = test_file_syntax()
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    if failed_imports:
        print(f" Missing packages: {', '.join(failed_imports)}")
