from setuptools import setup, find_packages
setup(name='dm-control', version='1.0.8', packages=find_packages(),
      include_package_data=True, install_requires=['mujoco==2.3.0', 'dm-env==1.6', 'dm-tree', 'lxml', 'absl-py', 'glfw', 'labmaze', 'pyopengl', 'scipy', 'requests'])
