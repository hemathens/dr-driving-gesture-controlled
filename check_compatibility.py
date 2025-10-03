import sys
import subprocess
import json
from urllib.request import urlopen

def check_package_compatibility(package_name, python_version):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = urlopen(url)
        data = json.loads(response.read())
        
        releases = data.get('releases', {})
        compatible_versions = []
        
        for version, files in releases.items():
            for file_info in files:
                requires_python = file_info.get('requires_python')
                if requires_python and f"{python_version[0]}.{python_version[1]}" in requires_python:
                    compatible_versions.append(version)
                    break
        
        return compatible_versions
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}\n")
    
    packages = [
        'numpy',
        'tensorflow',
        'tensorflowjs',
        'scikit-learn',
        'mediapipe',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'websockets'
    ]
    
    print("Checking package compatibility...\n")
    for package in packages:
        versions = check_package_compatibility(package, python_version)
        if versions:
            print(f"{package}: Compatible versions: {', '.join(versions[:3])}...")
        else:
            print(f"{package}: No compatible versions found")

if __name__ == "__main__":
    main()
