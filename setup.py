from distutils.core import setup

setup(
    name='pyku',
    version='0.1.0',
    packages=['pyku'],
    test_suite='test',
    url='',
    license='',
    author='Manuel Rota',
    author_email='',
    description='Utility to extract a sudoku from images',
    keywords='sudoku extract image',
    install_requires=[
        'numpy>=1.8.2',
        'scipy>=0.13.3',
        'opencv>=2.4.8'
    ],
    include_package_data=True,
    zip_safe=False
)