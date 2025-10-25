from setuptools import setup, find_packages

setup(
    name='cell_segmentation',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'fastapi',
        'uvicorn',
        'pillow',
        'scikit-learn',
        'numpy',
        'albumentations',
        'matplotlib',
        'seaborn',
        
    ],
    description='Library for cancer cell segmentation from MRI images',
)
