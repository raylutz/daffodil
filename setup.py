from setuptools import setup, find_packages

setup(
    name='Daffodil',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'xlsx2csv==0.8.2'
    ],
    author='Ray Lutz',
    author_email='raylutz@cognisys.com',
    description='Daffodil provides lightweight and fast 2-D dataframes',
    url='https://github.com/raylutz/Daffodil',
)
