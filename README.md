# lssbox
personal codes for Large-scale Structures measurement

## Installation
create a clean conda environment
```bash
conda create -n lssbox python=3.8
conda activate lssbox
```
### nbodykit
```bash
git clone http://github.com/bccp/nbodykit
cd nbodykit
conda install -c bccp --file requirements.txt "numpy<1.21"
conda install -c bccp --file requirements-extras.txt
pip install -e .
```
### pySpectrum
```bash
git clone https://github.com/zhaoruiyang98/pySpectrum
cd pySpectrum
conda install h5py
conda install -c conda-forge pyfftw
sudo apt install fftw3
pip install -e .
```
### lssbox
```bash
git clone https://github.com/zhaoruiyang98/lssbox.git
cd lssbox
conda install -c conda-forge --file requirements-test.txt
conda install --file requirements-dev.txt
pip install -e .
```