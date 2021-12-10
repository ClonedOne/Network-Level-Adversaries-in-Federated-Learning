import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='nlafl',
    version="1.0.0",
    author="Anonymous",
    author_email="Anonymous@anon.com",
    description="Network-Level Adversaries in Federated Learning Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nlafl/Network-Level-Adversaries-in-Federated-Learning",
    project_urls={
        "Bug Tracker": "https://github.com/nlafl/Network-Level-Adversaries-in-Federated-Learning/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: LINUX",
    ],
    install_requires=[
        'numpy==1.19.5',
        'scipy==1.4.1',
        'tensorflow==2.2.0',
        'pandas==1.3.3',
        'h5py==2.10.0',
        'celery==5.1.2',
        'fabric==2.6.0',
        'parse'
    ],
    package_dir={"": "src",'runParallel':'src/runParallel','tableCreation':'src/tableCreation'},
    packages=setuptools.find_packages(where="src") + setuptools.find_packages(where="src/tableCreation") + setuptools.find_packages(where="src/runParallel"),
    python_requires=">=3.8.10",
    entry_points={
        'console_scripts': ['nlafl=main:main']
    }

)