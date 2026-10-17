from setuptools import setup, find_packages
setup(name='star-dmc2gym', version='1.0.0', packages=find_packages(),
      include_package_data=True, install_requires=['gym==0.23.1', 'dm-env==1.6'])
