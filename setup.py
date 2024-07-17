from setuptools import find_packages,setup
from typing import List



def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file=file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    if("-e ." in requirements):
        requirements.remove("-e .")




setup(
name='GemStoneProject',
version='0.0.1',
author='Eshanth',
author_email='21ucs078@lnmiit.ac.in',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)