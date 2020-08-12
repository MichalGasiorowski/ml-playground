from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='titanic_module',
    version='0.1',
    author = 'Michal Gasiorowski',
    author_email = 'training-feedback@cloud.google.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Kaggle Titanic in Cloud ML',
    requires=[]
)