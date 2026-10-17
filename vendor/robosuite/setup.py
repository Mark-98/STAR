from setuptools import setup, find_packages
setup(name='robosuite', version='1.4.0', packages=find_packages(),
      include_package_data=True, install_requires=['mujoco==2.3.0', 'numpy', 'scipy', 'numba', 'pillow', 'termcolor'])
