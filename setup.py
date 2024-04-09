import setuptools
import pkg_resources

def read_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements

# Check for dependencies
try:
    pkg_resources.require(read_requirements())
except (ImportError, pkg_resources.DistributionNotFound) as e:
    print("Some dependencies are missing. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SegmentationApp",
    version="0.0.1",
    author="2404 Organ Segmentation",
    author_email="joey.xiang426@gmail.com",
    description="Proof of concept GUI app for medical image segmentation with inference models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2404-Organ-Segmentation/Segmentation_GUI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
