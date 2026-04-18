"""Setup script for bias_auditor package."""

from setuptools import setup, find_packages

setup(
    name="bias_auditor",
    version="0.1.0",
    description="Computer vision bias auditing system",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "open-clip-torch>=2.24.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "captum>=0.6.0",
        "shap>=0.42.0",
        "boto3>=1.26.0",
        "datasets>=2.14.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "gradio>=4.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
)
