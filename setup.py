#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取版本号
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'SR_py', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# 读取 README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="sisso-py",
    version=get_version(),
    author="SISSO-Py Team",
    author_email="3035326878@qq.com",
    description="Python implementation of Sure Independence Screening and Sparsifying Operator (SISSO)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sisso-py",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": [
            "sympy>=1.9.0",
            "joblib>=1.1.0",
            "numba>=0.56.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "sisso-py=sisso_py.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="symbolic regression, feature selection, SISSO, machine learning, physics",
)
