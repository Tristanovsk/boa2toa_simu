from setuptools import setup, find_packages


setup(
    name='OP3',
    version='1.0.0',
    packages=find_packages(exclude=['build']),
    package_data={'': ['*.so'],
    #     # If any package contains *.txt files, include them:
    #     '': ['*.txt'],
    #     'lut': ['data/lut/*.nc'],
    #     'aux': ['aux/*']
     },
    include_package_data=True,

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='Package to simulate plastic optical signal at the top-of-atmosphere level',
    # TODO update Dependent packages (distributions)
    install_requires=['pandas','cmocean','xarray','scipy', 'numpy',
                      'matplotlib'],

    entry_points={
        'console_scripts': [

        ]}
)