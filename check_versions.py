import sys
import pkg_resources

required_packages = [
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

print(f"Python version: {sys.version}\n")
print("Installed packages:")
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not installed")
