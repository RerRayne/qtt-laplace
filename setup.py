import pip

from setuptools import setup

try:
    import numpy as np
except ImportError:
    pip.main(["install", "--user", "numpy>=1.11.0"])

try:
    import cython
except ImportError:
    pip.main(["install", "--user", "cython>=0.24"])

setup(
    name='qtt_laplace',
    version='1.0',
    packages=['qtt_laplace', 'qtt_laplace.basis', 'qtt_laplace.basis.linear_basis'],
    url='https://github.com/RerRayne/qtt-laplace.git',
    author='Markeeva L.',
    author_email='rerayne@gmail.com',
    description='A library for solving elliptic PDE using the Quantized Tensor Train Decomposition (QTT).',
    install_requires=["six>=1.10.0", "ttpy>=1.2.0"]
)
