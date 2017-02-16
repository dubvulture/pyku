from distutils.core import setup

setup(
    name='pyku',
    version='0.1.0',
    packages=['pyku'],
    license='LICENSE.txt',
    author='Manuel Rota',
    description='Utility to extract a sudoku from images',
    keywords='sudoku extract image',
    install_requires=[
        'numpy>=1.8.2',
        'scipy>=0.13.3',
        'opencv>=2.4.8'
        ],
    include_package_data = True
)
