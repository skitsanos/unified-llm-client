#!/usr/bin/env python
"""
Build script for the unified-llm-client package

@author: skitsanos
"""

import os
import shutil
import subprocess
import sys


def clean_build_dirs():
    """Clean up build directories"""
    print("Cleaning build directories...")
    dirs_to_clean = ["dist", "build", "llm_client.egg-info", "skitsanos_llm_client.egg-info",
                     "unified_llm_client.egg-info"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")


def build_package():
    """Build the package"""
    print("Building package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    subprocess.check_call([sys.executable, "-m", "build", "--sdist", "--wheel", "."])


def check_package():
    """Check the built package with twine"""
    print("Checking package...")
    subprocess.check_call([sys.executable, "-m", "twine", "check", "dist/*"])


def upload_package(test=True):
    """Upload the package to PyPI or TestPyPI"""
    if test:
        print("Uploading to TestPyPI...")
        subprocess.check_call([
            sys.executable, "-m", "twine", "upload", "--repository-url", "https://test.pypi.org/legacy/", "dist/*"
        ])
    else:
        print("Uploading to PyPI...")
        subprocess.check_call([sys.executable, "-m", "twine", "upload", "dist/*"])


if __name__ == "__main__":
    clean_build_dirs()
    build_package()
    check_package()
    
    # Comment out or uncomment based on your needs
    # upload_package(test=True)  # Upload to TestPyPI
    # upload_package(test=False)  # Upload to PyPI
    
    print("\nBuild and check completed successfully!")
    print("To upload to PyPI, use the following commands:")
    print("  python build_package.py")
    print("  python -m twine upload dist/*")
    print("\nTo upload to TestPyPI first, use:")
    print("  python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*")
