# ED_cPSD

This repository is dedicated to the simulation of continuous phase-size distributions based on the sequential Erosion-Dilation continuous Phase-Size Distribution (ED-cPSD) method. This novel approach combines the calculation of an Euclidean Distance Map (EDM) via the Meijster<sup>[[1]](http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf)</sup> algorithm for fast erosion-dilation operations at increasing radii. The individual phase-spaces obfuscated by the erosion step at each radii will then not return to the original size and shape via the dilation operation. These changes at each radii can be quantified, and then when normalized by the total number of phase-voxels removed at each step, give a continuous (in discrete space) probability density for phase-size, which is interpreted as being the phase-size distribution.

The [Documentation](#documentation) and [publications](#publications) below contain more detail about the computational model, validation, and some example use cases for this code.

# Table of Contents

1. [Requirements](#requirements)
2. [Compilation](#compilation)
3. [Required Files](#required-files)
4. [Publications](#ppublications)
5. [Authors](#code-authors)
6. [Documentation](#documentation)
7. [Acknowledgements](#acknowledgements)
9. [References](#references)

# Requirements

The base source code only requires the following:
- C++17 or newer.
- GNU Compiler (any recent version, was tested on gcc 13.1.0, but other versions should be fine).
- OpenMP (any recent version, this is attached to the gcc compiler on Windows).
- [stb_image](https://github.com/nothings/stb) any recent version.

The GUI requires:
- Qt6.8 or newer (older versions might work, not guaranteed).

# Compilation

To compile the base code without the GUI, simply run the following:

```bash
g++ -fopenmp ./ED_PSD_CPU.cpp -o anyName
```

That will create the executable `anyName.exe` on Windows and `anyName.out` on Linux.

To compile the GUI into an executable, CMAKE is necessary.

# Publications

If you produce publishable results using this package, please acknowledge the following publication

- (Software publication incoming).

Additionally, you may want to consider acknowledging other publications that used this work:

- Adam, A., Wang, F., & Li, X. (2022). Efficient reconstruction and validation of heterogeneous microstructures for energy applications. International Journal of Energy Research. https://doi.org/10.1002/er.8578

# Authors
- Main developer: Andre Adam (The University of Kansas)
    - [ResearchGate](https://www.researchgate.net/profile/Andre-Adam-2)
    - [GoogleScholar](https://scholar.google.com/citations?hl=en&user=aP_rDkMAAAAJ)
    - [GitHub](https://github.com/adama-wzr)
- Advisor: Dr. Xianglin Li (Washingtion University in St. Louis)
    - [Website](https://xianglinli.wixsite.com/mysite)
    - [GoogleScholar](https://scholar.google.com/citations?user=8y0Vd8cAAAAJ&hl=en)
- Advisor: Dr. Guang Yang (Oak-Ridge National Laboratory)
    - [Website](https://www.ornl.gov/staff-profile/guang-yang)
    - [GoogleScholar](https://scholar.google.com/citations?user=Ph_5mDMAAAAJ&hl=en)
- Advisor: Dr. Huazhen Fang (University of Kansas)
    - [Website](https://fang.ku.edu/)
    - [Lab Website](https://www.issl.space/)
    - [GoogleScholar](https://scholar.google.com/citations?user=3m7Yd4YAAAAJ&hl=en)
# Documentation

# Acknowledgements

This work used Expanse(GPU)/Bridges2(CPU) at SDSC/PSC through allocations MAT210014 and MAT230071 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services \& Support (ACCESS) program, which is supported by National Science Foundation grants 2138259, 2138286, 2138307, 2137603, and 2138296<sup>[[2]](https://doi.org/10.1145/3569951.3597559)</sup>.

X.L. highly appreciates the support from the National Science Foundation (Award 1941083 and 2329821).

The information, data, or work presented herein was funded in part by the Advanced Research Projects Agency-Energy (ARPA-E), U.S. Department of Energy, under Award Number 18/CJ000/08/08.

# References
1. Meijster, A., Roerdink, J. B. T. M., & Hesselink, W. H. (n.d.). A General Algorithm for Computing Distance Transforms in Linear Time. In Computational Imaging and Vision (pp. 331–340). Kluwer Academic Publishers. https://doi.org/10.1007/0-306-47025-x_36
2. Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns. 2023. ACCESS: Advancing Innovation: NSF’s Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support. “In Practice and Experience in Advanced Research Computing (PEARC ’23)”, July 23–27, 2023, Portland, OR, USA. ACM, New York, NY, USA, 4 pages. [https://doi.org/10.1145/3569951.3597559](https://doi.org/10.1145/3569951.3597559).
