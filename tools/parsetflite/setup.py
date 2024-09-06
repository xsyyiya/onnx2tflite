#!/usr/bin/env python
# coding=utf-8
import os
from setuptools import setup, find_packages
abs_path = os.path.dirname(os.path.abspath(__file__))

setup(
        name="parsetflite",
        version="1.0",
        author="xsy",
        description="parsing tflite model",
        # long_description=open(os.path.join(abs_path, "readme.md")).read(),
        # long_description_content_type='text/markdown',
        # packages=find_packages(include=['parsetflite']),
        # license="Apache-2.0",
        # platforms=["Windows", "linux"],
        # install_requires=open(os.path.join(abs_path, "requirements.txt")).read().splitlines()
        py_modules = ['parsetflite'],
        entry_points={'console_scripts': ['parse-lite = parsetflite:main']}
    
)
