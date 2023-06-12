# 1D Inverse Probability Integral Transform

Quick tool for mapping $1$-dimensional probability density functions (PDFs) or cumulative distribution functions (CDFs) to a fiducial PDF or CDF. This can be fitted from either the fiducial PDF or CDF. The method works by inverting the CDF of the fiducial distribution and then evaluating the inverse CDF at the CDF of the test distribution.


## Example

```python
import numpy
from scipy.stats import norm

from inv_pit import InversePIT


fid_dist = norm(loc=0, scale=1)
test_dist = norm(loc=0.5, scale=0.5)

x = numpy.linspace(-4, 4, 1000)
fid_pdf = fid_dist.pdf(x)
test_pdf = test_dist.pdf(x)

mapper.fit_from_pdf(x, fid_pdf)
trans_x, transf_pdf = mapper.transform_pdf(x, test_pdf)
```
For a more concrete example, see the [example script](https://github.com/Richard-Sti/inv_pit/blob/master/example.py), which generates the following plot. In the left column, the transform is fitted from the PDF, whereas in the right column it is fitted directly from the CDF.


![alt text](https://github.com/Richard-Sti/inv_pit/blob/master/example.png?raw=true)

## Installation

The code can be manually installed by cloning the repository,
```bash
git clone git@github.com:Richard-Sti/inv_pit.git
```
and then creating a virtual environment and installing it in it.
```bash
python3 -m venv venv_invpit
source venv_invpit/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
python -m pip install .
```
which will install the package in the virtual environment ``venv_invpit``.
