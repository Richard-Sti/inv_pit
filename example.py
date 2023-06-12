# Copyright (C) 2023 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Example script to illustrate the usage of the InversePIT class. We map a `test`
distribution to a `fiducial` distribution.
"""
import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from scipy.stats import norm

from inv_pit import InversePIT


fid_dist = norm(loc=0, scale=1)
test_dist = norm(loc=0.5, scale=0.5)

x = numpy.linspace(-4, 4, 1000)
fid_pdf = fid_dist.pdf(x)
test_pdf = test_dist.pdf(x)

fid_cdf = fid_dist.cdf(x)
test_cdf = test_dist.cdf(x)

with plt.style.context("science"):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(figsize=(3.5 * 1.5, 2.625 * 1.25),
                            nrows=2, ncols=2,
                            sharex=True, sharey="row")
    fig.subplots_adjust(hspace=0, wspace=0)

    mapper = InversePIT()
    for i in range(2):
        if i == 0:
            mapper.fit_from_cdf(x, fid_cdf)
        else:
            mapper.fit_from_pdf(x, fid_pdf)

        axs[0, i].plot(x, fid_pdf, label="Fiducial")
        axs[0, i].plot(x, test_pdf, label="Test")
        axs[0, i].plot(*mapper.transform_pdf(x, test_pdf), ls="--",
                       label="Transformed")

        axs[1, i].plot(x, fid_cdf)
        axs[1, i].plot(x, test_cdf)
        axs[1, i].plot(*mapper.transform_cdf(x, test_cdf), ls="--")

        offset = 1e-5
        axs[0, i].set_xlim(-4 + offset, 4 - offset)
        axs[1, i].set_ylim(0, 1 - offset)
        axs[0, i].set_ylim(offset)
        axs[1, i].set_xlabel(r"$x$")

    axs[0, 0].set_ylabel(r"$\mathrm{PDF}(x)$")
    axs[1, 0].set_ylabel(r"$\mathrm{CDF}(x)$")

    axs[0, 0].set_title("Fitting from the PDF")
    axs[0, 1].set_title("Fitting from the CDF")
    axs[0, 0].legend(fontsize="small")

    fig.tight_layout(h_pad=0, w_pad=0)
    fig.savefig("example.png", dpi=450, bbox_inches="tight")
    plt.close()
