from setuptools import setup, find_packages


# install requirements
inst_reqs = [
'numpy',
]

setup(
    name = 'SpatialNets',
    version = '0.2',

    packages = ['SpatialNets'],
    include_package_data = False,
    plantforms = 'any',
    install_requires=inst_reqs,
    package_data = {'SpatialNets/model': ['*.pth'],'SpatialNets_pavia':['*.py']},
    scripts = ['SpatialNets_pavia/model.py']
)