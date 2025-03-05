# IPComp: Interpolation-Based Progressive Lossy Compression for Scientific Datasets
## Overview
IPComp is an interpolation-prediction-based lossy compressor designed specifically for scientific datasets. Its key feature is progressive compression, enabling multiple levels of decompression precision from a single compressed file. This means that higher-fidelity data can be reconstructed incrementally, requiring only additional data to be loaded rather than reloading the entire dataset.

Even in non-progressive compression scenarios, where a single fixed precision is required, IPComp achieves high compression ratios, making it an efficient choice for lossy data reduction.

## Key Features
 - Progressive Decompression: A single compressed file supports multiple precision levels, reducing storage and computation overhead for adaptive applications.
 - Incremental Loading: Higher-fidelity reconstructions require only additional data, minimizing I/O costs.
 - High Compression Efficiency: Even in standard, non-progressive compression use cases, IPComp achieves competitive compression ratios.
## Usage
Command-line Syntax:
```
IPComp <datafile> -dataType[f/d] -dim_num <dim1> <dim2> ... -bound_mode -bound_num <bound1> <bound2> ...
```
Example Commands: 
- Error-Bounded Progressive Compression
```
./src/IPComp density.d64 -d -3 256 384 384 -error -3 1e-3 1e-4 1e-5
```
This compresses the dataset density.d64 with progressive error bounds at 1e-3, 1e-4, and 1e-5.

- Bitrate-Bounded Progressive Compression
```
./src/IPComp density.d64 -d -3 256 384 384 -bitrate -3 1.0 2.0 3.0 
```
This compresses density.d64 using bitrate constraints of 1.0, 2.0 and 3.0 bits per value.
## License

IPComp is licensed under the BSD 3-Clause License.

This software is based on SZ (Version 3.0), originally developed by Argonne National Laboratory.

Original SZ Authors: Sheng Di, Kai Zhao, Xin Liang, Dingwen Tao, Franck Cappello  
Copyright © 2016, UChicago Argonne, LLC  

## Installation
```
mkdir build && cd build
cmake ..
make
```
Then, you'll find all the executables in [INSTALL_DIR]/test
