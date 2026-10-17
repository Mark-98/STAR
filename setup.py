"""Compatibility metadata for the legacy Python 3.8/setuptools runtime."""
from setuptools import find_packages, setup

setup(name='star-rl', version='0.1.0', packages=find_packages(include=['star', 'star.*', 'diffusion_policy', 'diffusion_policy.*']),
      python_requires='>=3.8,<3.11', install_requires=['PyYAML>=6,<7'],
      entry_points={'console_scripts': ['star-train=star.training.rl:main',
                                       'star-eval=star.evaluation.rl:main',
                                       'star-train-diffusion=star.training.diffusion:main',
                                       'star-eval-diffusion=star.evaluation.diffusion:main']})
