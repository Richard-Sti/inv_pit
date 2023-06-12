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
Inverse probability integral transform code.
"""
import numpy
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


class InversePIT:
    """
    Inverse probability integral transform of one 1-dimensional distribution
    onto another. The transformation is trained on a fiducial distribution to
    which any other distribution can then be mapped.
    """
    _cdf_map = None

    def transform_pdf(self, x, pdf):
        """
        Transform a test PDF to the fiducial PDF used to fit the model.

        Parameters
        ----------
        x : 1-dimensional array
            Values where the PDF is evaluated.
        pdf : 1-dimensional array
            PDF values at `x`.

        Returns
        -------
        x : 1-dimensional array
            Values where the transformed PDF is evaluated.
        pdf : 1-dimensional array
            Transformed PDF values at `x`.
        """
        # We first transform the PDF to a CDF, then transform the CDF, and
        # then transform back to a PDF.
        cdf = cumulative_trapezoid(pdf, x, initial=0)
        cdf /= cdf[-1]
        x, cdf = self.transform_cdf(x, cdf)
        # There will warnings when `x` is not changing because of
        # the inverse CDF mapping. Those points are set to NaN.
        with numpy.errstate(all="ignore"):
            pdf = numpy.gradient(cdf, x, edge_order=2)
        return x, pdf

    def transform_cdf(self, x, cdf):
        """
        Transform a test CDF to the fiducial CDF used to fit the model.

        Parameters
        ----------
        x : 1-dimensional array
            Values where the CDF is evaluated.
        cdf : 1-dimensional array
            CDF values at `x`.

        Returns
        -------
        x : 1-dimensional array
            Values where the transformed CDF is evaluated.
        cdf : 1-dimensional array
            Transformed CDF values at `x`.
        """
        if self._cdf_map is None:
            raise RuntimeError("Transformation must be fitted first!")
        x = self._cdf_map(cdf)
        return x, cdf

    def fit_from_cdf(self, x, cdf, interp_kind="linear"):
        """
        Fit the transformation of any PDF to the fiducal PDF (or CDF). Fitted
        from the CDF, not the PDF.

        Parameters
        ----------
        x : 1-dimensional array
            Values where the CDF is evaluated.
        cdf : 1-dimensional array
            CDF values at `x`.
        interp_kind : str, optional
            Interpolation kind used for the inverse CDF mapping. See
            `scipy.interpolate.interp1d` for details.

        Returns
        -------
        None
        """
        assert x.ndim == 1 and x.shape == cdf.shape
        # Optinally enforce normalisation.
        if not numpy.isclose(cdf[-1], 1):
            cdf /= cdf[-1]
        self._cdf_map = interp1d(cdf, x, kind=interp_kind, bounds_error=False,
                                 fill_value="extrapolate")

    def fit_from_pdf(self, x, pdf, interp_kind="linear"):
        """
        Fit  the transformation of any PDF to the fiducal PDF (or CDF). Fitted
        from the PDF, which is used to calculate the CDF.

        Parameters
        ----------
        x : 1-dimensional array
            Values where the PDF is evaluated.
        pdf : 1-dimensional array
            PDF values at `x`.
        interp_kind : str, optional
            Interpolation kind used for the inverse CDF mapping. See
            `scipy.interpolate.interp1d` for details.

        Returns
        -------
        None
        """
        assert x.ndim == 1 and x.shape == pdf.shape
        # Compute CDF via cumulative trapezoid rule and enforce normalisation.
        cdf = cumulative_trapezoid(pdf, x, initial=0)
        cdf /= cdf[-1]
        self.fit_from_cdf(x, cdf, interp_kind=interp_kind)
