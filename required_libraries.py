import sys
import subprocess

# List of packages to install
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "seaborn",
    "tensorflow",
    "torch",  # Correct package name for PyTorch
    "pymongo",
    "dnspython",
    "wfdb"
]

def install_packages(package_list):
    for package in package_list:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")

if __name__ == "__main__":
    install_packages(packages)
    print("All packages processed.")
