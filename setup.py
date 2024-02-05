from setuptools import setup, find_packages

setup(
    name='Pydf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'xlsx2csv==0.8.2'
    ],
    author='Ray Lutz',
    author_email='raylutz@cognisys.com',
    description='Python Daffodil (Pydf) provides lightweight and fast 2-D dataframes',
    url='https://github.com/raylutz/Pydf',
)
