#!/usr/bin/env python3
"""
Setup script for AI-Based MRI Image Classification for Brain Tumor Categorization
"""

from setuptools import setup, find_packages

# Safely read README and requirements
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "AI-Based MRI Image Classification for Brain Tumor Categorization"

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = fh.read().splitlines()
except FileNotFoundError:
    requirements = ["tensorflow", "opencv-python", "numpy", "matplotlib", "scikit-learn", "seaborn"]

setup(
    name="brain-tumor-categorization", # Updated name
    version="1.1.0",
    author="Susan Aryal", # Updated from your synopsis
    author_email="susanaryal089@gmail.com.com",
    description="AI-Based MRI Image Classification for 4-Class Brain Tumor Categorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arsusan/Brain-tumor-detection-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            # These point to the 'main' function inside your train.py
            "train-tumor-model=train:run_training", 
            "predict-tumor=predict:main",
        ],
    },
)