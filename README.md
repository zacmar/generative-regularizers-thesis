# Generative Models as Regularizers for Inverse Problems in Imaging

This thesis encompasses largely the two articles:

- **"Stable Deep MRI Reconstruction using Generative Priors"** 
  - [Code](https://github.com/VLOGroup/stable-deep-mri)
  - [Article](https://ieeexplore.ieee.org/document/10237244)

- **"Product of Gaussian Mixture Diffusion Models"** 
  - [Code](https://github.com/VLOGroup/PoGMDM)
  - [Article](https://link.springer.com/article/10.1007/s10851-024-01180-3)

The source code needs to be compiled with `lualatex` and uses the tikz externalization library for parallel processing and caching of figures.
Thus, compilation looks something like this:

```bash
# first latexmk run, populates `main.makefile` with code to produce figures
latexmk -shell-escape -lualatex main.tex 
# make figures
make -j -f main.makefile
# second latexmk run
latexmk -shell-escape -lualatex main.tex 
```
