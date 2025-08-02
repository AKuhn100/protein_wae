"""
Setup script for Protein WAE package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="protein-wae",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Wasserstein Autoencoder for protein sequence generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/protein-wae",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "protein-wae-train=protein_wae.scripts.train:main",
            "protein-wae-sample=protein_wae.scripts.sample:main",
        ],
    },
)