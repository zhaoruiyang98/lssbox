from setuptools import setup, find_packages
from pathlib import Path


def read_long_description():
    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        text = f.read()
    return text


exclude_dir = ['docs', 'tests']
packages = find_packages(exclude=exclude_dir)

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Typing :: Typed",
]

with Path('requirements.txt').open() as f:
    dependencies = f.readlines()

with Path('requirements-test.txt').open() as f:
    test_reqs = f.readlines()

with Path('requirements-dev.txt').open() as f:
    dev_reqs = f.readlines()
dev_reqs += test_reqs

setup(
    name='lssbox',
    version='0.0.1',
    long_description=read_long_description(),
    url='https://github.com/zhaoruiyang98/lssbox',
    project_urls={
        'Source': 'https://github.com/zhaoruiyang98/lssbox',
        'Tracker': 'https://github.com/zhaoruiyang98/lssbox/issues',
        'Licensing': 'https://github.com/zhaoruiyang98/lssbox/blob/main/LICENSE'
    },
    author='Ruiyang Zhao',
    author_email='zhaoruiyang19@mails.ucas.edu.cn',
    license='GPLv3',
    python_requires='>=3.8, <3.9',
    keywords='cosmology large-scale-structure',
    packages=packages,
    install_requires=dependencies,
    extras_require={
        'test': test_reqs,
        'dev': dev_reqs,
    },
    package_data={
        'lssbox': ['py.typed']
    },
    classifiers=classifiers,
    zip_safe=False,
)
