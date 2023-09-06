# lssbox
personal codes for Large-scale Structures measurement

## Installation
create a clean conda environment
```bash
conda create -n lssbox python=3.8
conda activate lssbox
```

If you are using apple M1/M2
```bash
CONDA_SUBDIR=osx-64 conda create -n lssbox python=3.8
conda activate lssbox
conda config --env --set subdir osx-64
```
### nbodykit
If you have set `channel_priority: strict` in your `.condarc` file, please comment out it in this section.
```bash
git clone https://github.com/bccp/nbodykit
cd nbodykit
conda install -c bccp --file requirements.txt "numpy<1.21"
conda install -c bccp --file requirements-extras.txt
pip install -e .
```
### lssbox
```bash
git clone https://github.com/zhaoruiyang98/lssbox.git
cd lssbox
# (extra dependencies, optional)
# conda install -c conda-forge --file requirements-test.txt
# conda install -c conda-forge --file requirements-dev.txt
pip install -e .
```
### pySpectrum
optional
```bash
git clone https://github.com/zhaoruiyang98/pySpectrum
cd pySpectrum
conda install h5py
conda install -c conda-forge pyfftw
sudo apt install fftw3
pip install -e .
```