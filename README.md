# Why summation by parts is not enough

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18596276.svg)](https://doi.org/10.5281/zenodo.18596276)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{glaubitz2026summation,
  title={Why summation by parts is not enough},
  author={Glaubitz, Jan and Iske, Armin and Lampert, Joshua and Öffner, Philipp},
  year={2026},
  month={02},
  eprint={2602.10786},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{glaubitz2026summationRepro,
  title={Reproducibility repository for
         ``{W}hy summation by parts is not enough"},
  author={Glaubitz, Jan and Iske, Armin and Lampert, Joshua and Öffner, Philipp},
  year={2026},
  howpublished={\url{https://github.com/JoshuaLampert/2026\_SBP\_not\_enough}},
  doi={10.5281/zenodo.18596276}
}
```

## Abstract

We investigate the construction and performance of summation-by-parts (SBP) operators, which offer a powerful
framework for the systematic development of structure-preserving numerical discretizations of partial differential
equations. Previous approaches for the construction of SBP operators have usually relied on either local methods
or sparse differentiation matrices, as commonly used in finite difference schemes. However, these methods often
impose implicit requirements that are not part of the formal SBP definition. We demonstrate that adherence
to the SBP definition alone does not guarantee the desired accuracy, and we identify conditions for SBP operators
to achieve both accuracy and stability. Specifically, we analyze the error minimization for an augmented
basis, discuss the role of sparsity, and examine the importance of nullspace consistency in the construction of
SBP operators. Furthermore, we show how these design criteria can be integrated into a recently proposed
optimization-based construction procedure for function space SBP (FSBP) operators on arbitrary grids. Our
findings are supported by numerical experiments that illustrate the improved accuracy for the numerical solution
using the proposed SBP operators.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.12.4.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- Jan Glaubitz (Linköping University, Sweden)
- Armin Iske (University of Hamburg, Germany)
- Joshua Lampert (University of Hamburg, Germany)
- Philipp Öffner (TU Clausthal, Germany)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
