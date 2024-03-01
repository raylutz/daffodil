from setuptools import setup, find_packages

setup(
    name='daffodil',
    version='0.2.2',
    packages=find_packages(),
    install_requires=[
        'xlsx2csv==0.8.2'
    ],
    author='Ray Lutz',
    author_email='raylutz@cognisys.com',
    description='Daffodil provides lightweight and fast 2-D dataframes',
    url='https://github.com/raylutz/daffodil',
)
