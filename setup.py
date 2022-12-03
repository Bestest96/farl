from setuptools import setup, find_packages

setup(
    name='farl',
    version='0.1.0',
    url='https://github.com/skalermo/farl',
    author='skalermo',
    author_email='skalermo@gmail.com',
    description='Implementation of Function Approximation Reinforcement Learning algorithm',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
    ],
)
