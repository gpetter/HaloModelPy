from setuptools import setup
from setuptools import find_packages

setup(
    name='halomodelpy', #project name
    version='0.1.0',
    description='calculate HM observables',
    #url
    author='Petter',
    author_email='Grayson.C.Petter.GR@dartmouth.edu', #not a real e-mail
    license='',
    packages=find_packages(),
    install_requires=['colossus', 'mcfit']
)