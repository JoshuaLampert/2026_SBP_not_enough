# Beyond the summation-by-parts property: nullspace consistency, sparsity, and regularization for FSBP operators

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18596276.svg)](https://doi.org/10.5281/zenodo.18596276)

This repository contains information and code to reproduce the results presented in the
article (previously titled "Why summation by parts is not enough")
```bibtex
@online{glaubitz2026summation,
  title={Beyond the summation-by-parts property: nullspace consistency, sparsity, and
         regularization for {FSBP} operators},
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
         ``{B}eyond the summation-by-parts property: nullspace consistency, sparsity,
         and regularization for {FSBP} operators"},
  author={Glaubitz, Jan and Iske, Armin and Lampert, Joshua and Öffner, Philipp},
  year={2026},
  howpublished={\url{https://github.com/JoshuaLampert/2026\_SBP\_not\_enough}},
  doi={10.5281/zenodo.18596276}
}
```

## Abstract

We investigate the construction and performance of summation-by-parts (SBP) operators, which offer a powerful
framework for the systematic development of structure-preserving numerical discretizations of partial differential
equations. Previous approaches for the construction of SBP operators have usually relied on either local methods or
sparse differentiation matrices, as commonly used in finite difference schemes. However, these methods often impose
implicit requirements that are not part of the formal SBP definition. We demonstrate that adherence to the SBP
definition alone does not guarantee the desired accuracy, and we make additional conditions explicit that SBP
operators need to satisfy in order to achieve accuracy. While these conditions are known in the SBP literature,
they are usually enforced only implicitly by the respective construction procedure. Specifically, we analyze the
error minimization for an augmented basis, discuss the role of sparsity, and examine the importance of nullspace
consistency in the construction of SBP operators. A dispersion and dissipation analysis shows that the loss of
accuracy has two sources. First, a lack of nullspace consistency produces stationary modes, which are neither
transported nor damped by the scheme and whose contribution to the error converges more slowly under mesh
refinement. Second, the physical mode, i.e., the discrete approximation of an exact traveling wave, can be poorly
resolved, and the resulting error dominates in long-time simulations. Furthermore, we show how these design
criteria can be integrated into a recently proposed optimization-based construction procedure for function space
SBP (FSBP) operators on arbitrary grids. Our findings are supported by numerical experiments that illustrate the
improved accuracy of the numerical solutions obtained with the proposed SBP operators.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). The numerical experiments presented
in this article were performed using Julia v1.13.1.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- Jan Glaubitz (Linköping University, Sweden)
- Armin Iske (University of Hamburg, Germany)
- Joshua Lampert (University of Hamburg, Germany)
- Philipp Öffner (Clausthal University of Technology, Germany)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
