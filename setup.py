from setuptools import setup, find_packages


# install requirements
inst_reqs = [
'numpy',
'pytorch',
'tifffile',
'torchvision'
]

setup(
    name = 'SpatialNets_pavia',
    version = '0.1',

    packages = ['SpatialNets_pavia'],
    include_package_data = False,
    plantforms = 'any',
    install_requires=inst_reqs,
    package_data = {'SpatialNets/model': ['*.pth'],'SpatialNets_pavia':['*.py']},
    scripts = ['SpatialNets_pavia/model.py']
)