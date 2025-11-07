"""Setup script for DHC-ASAGI Hybrid System."""

from setuptools import setup, find_packages

setup(
    name="dhc-asagi-hybrid",
    version="1.0.0",
    author="Sung Hun Kwag",
    description="Hybrid architecture combining DHC-SSM and ASAGI",
    long_description="A hybrid AI system integrating DHC-SSM (State Space Model) and ASAGI (Autonomous Self-Organizing AI) with four integration modes.",
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/DHC-ASAGI-Hybrid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
