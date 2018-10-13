from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'h5py',
    'numpy==1.14.3',
    'pydicom==1.1.0',
    'tensorflow==1.10.0',
    'scipy==1.1.0',
    'setuptools==39.1.0',
    'opencv_python==3.4.3.18',
    'Keras==2.2.2',
    'pandas==0.23.0',
    'scikit_learn==0.20.0',
    'cloudstorage'
    ]

setup(
    name='FasterRCNN',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='frcnn on rsna.'
)