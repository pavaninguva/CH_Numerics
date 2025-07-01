import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CloughTocher2DInterpolator



"""
For the classic interpolation on 2d
"""


class Pchip2D:
    def __init__(self, x, y, Z):
        Z = np.asarray(Z)
        if not np.isfinite(Z).all():
            bad = np.where(~np.isfinite(Z))
            raise ValueError(f"Non-finite entries in Z at indices {bad}: {Z[bad]}")
        self.Z = Z
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        # build one 1D PCHIP in x for each fixed y[j]
        self._x_splines = [PchipInterpolator(self.x, Z[:, j])
                           for j in range(len(self.y))]

    def __call__(self, xq, yq):
        # ensure numpy arrays
        xq_arr = np.asarray(xq)
        yq_arr = np.asarray(yq)
        xq_arr = np.clip(xq_arr, self.x[0], self.x[-1])
        yq_arr = np.clip(yq_arr, self.y[0], self.y[-1])
        shape  = xq_arr.shape

        # flatten
        xf = xq_arr.ravel()
        yf = yq_arr.ravel()

        # prepare output
        zf = np.empty_like(xf, dtype=float)

        # find unique x's so we only build each row-spline once
        ux, inv = np.unique(xf, return_inverse=True)

        for i, xval in enumerate(ux):
            # 1) evaluate all row-splines at xval → a 1D array Wj
            Wj = np.array([s(xval) for s in self._x_splines])
            # 2) build one PCHIP in y
            ysp = PchipInterpolator(self.y, Wj)
            # find all positions where xf == this xval
            mask = (inv == i)
            # 3) evaluate y-spline over those y's
            zf[mask] = ysp(yf[mask])

        # reshape back
        return zf.reshape(shape)

    def partial_x(self, xq, yq):
        # same flatten/unique trick
        xq_arr = np.asarray(xq); yq_arr = np.asarray(yq)
        shape  = xq_arr.shape
        xf, yf = xq_arr.ravel(), yq_arr.ravel()
        dxf = np.empty_like(xf, dtype=float)

        ux, inv = np.unique(xf, return_inverse=True)
        for i, xval in enumerate(ux):
            # 1) ∂/∂x of each row-spline at xval
            dW = np.array([s.derivative()(xval) for s in self._x_splines])
            # 2) build PCHIP in y on these slopes
            ysp = PchipInterpolator(self.y, dW)
            mask = (inv == i)
            dxf[mask] = ysp(yf[mask])

        return dxf.reshape(shape)

    def partial_y(self, xq, yq):
        xq_arr = np.asarray(xq); yq_arr = np.asarray(yq)
        shape  = xq_arr.shape
        xf, yf = xq_arr.ravel(), yq_arr.ravel()
        dyf = np.empty_like(xf, dtype=float)

        ux, inv = np.unique(xf, return_inverse=True)
        for i, xval in enumerate(ux):
            # 1) compute Wj at xval
            Wj = np.array([s(xval) for s in self._x_splines])
            # 2) build y-spline and take its derivative
            ysp = PchipInterpolator(self.y, Wj).derivative()
            mask = (inv == i)
            dyf[mask] = ysp(yf[mask])

        return dyf.reshape(shape)
    

def generate_spline_dfdphi1(chi12, chi13, N1, knots):
    """
    chi12, chi13: interaction parameters
    N1          : parameter in derivative
    knots       : number of uniform knots per dimension
    returns     : callable spline phi1,phi2 -> dfdphi1(phi1,phi2)
    """
    domain_min, domain_max = 1e-16, 1 - 1e-16

    # the "true" 2D derivative function
    def dfdphi1(phi1, phi2):
        return (1/N1)*np.log(phi1) + (1/N1) \
               + chi12*phi2 \
               + chi13*(1 - phi1 - phi2)

    # build uniform grid of knots
    phi1_knots = np.linspace(domain_min, domain_max, knots)
    phi2_knots = np.linspace(domain_min, domain_max, knots)
    P1, P2 = np.meshgrid(phi1_knots, phi2_knots, indexing='ij')
    Z = dfdphi1(P1, P2)

    phi1_knots_spline = np.linspace(0,1,knots)
    phi2_knots_spline = np.linspace(0,1,knots)
    

    # build and return the tensor‐product PCHIP
    spline2d = Pchip2D(phi1_knots_spline, phi2_knots_spline, Z)
    return spline2d

def generate_spline_dfdphi2(chi12, chi23, N2, knots):
    """
    chi12, chi13: interaction parameters
    N1          : parameter in derivative
    knots       : number of uniform knots per dimension
    returns     : callable spline phi1,phi2 -> dfdphi1(phi1,phi2)
    """
    domain_min, domain_max = 1e-16, 1 - 1e-16

    # the "true" 2D derivative function
    def dfdphi2(phi1, phi2):
        return (1/N2)*np.log(phi2) + (1/N2) \
               + chi12*phi1 \
               + chi23*(1 - phi1 - phi2)

    # build uniform grid of knots
    phi1_knots = np.linspace(domain_min, domain_max, knots)
    phi2_knots = np.linspace(domain_min, domain_max, knots)
    P1, P2 = np.meshgrid(phi1_knots, phi2_knots, indexing='ij')
    Z = dfdphi2(P1, P2)

    phi1_knots_spline = np.linspace(0,1,knots)
    phi2_knots_spline = np.linspace(0,1,knots)

    # build and return the tensor‐product PCHIP
    spline2d = Pchip2D(phi1_knots_spline, phi2_knots_spline, Z)
    return spline2d

"""
Scattered Interpolation in 2D for dfdphi3

"""

def generate_spline_dfdphi3(chi13,chi23,N3,knots, return_knots=False):
    """
    chi13, chi23: interaction parameters
    N3          : parameter in derivative
    knots       : number of uniform knots per dimension
    returns     : callable spline phi1,phi2 -> dfdphi3(phi1,phi2)
    """
    # the "true" 2D derivative function
    def dfdphi3(phi1, phi2):
        return (1/N3)*np.log(1-phi1-phi2) + (1/N3) \
               + chi13*phi1 \
               + chi23*phi2
    
    #Knots for interior func eval
    phi1_knots = np.linspace(1e-16,1-1e-16,knots)
    phi2_knots = np.linspace(1e-16,1-1e-16,knots)

    P1, P2 = np.meshgrid(phi1_knots,phi2_knots)
    mask = (P1 + P2 < 1)
    phi1_int = P1[mask]
    phi2_int = P2[mask]

    dfdc_int = dfdphi3(phi1_int,phi2_int)

    phi1_b = np.linspace(1e-15,1-1e-15,knots)
    phi2_b = 1 - phi1_b

    phi3_b = 1 - phi1_b - phi2_b
    phi3_b = np.clip(phi3_b,1e-16,None)

    dfdc_b = (
        (1/N3)*np.log(phi3_b)
        +(1/N3)
        +chi13*phi1_b
        +chi23*phi2_b
    )

    #Set up spline bits
    phi1_int_ = np.where(np.isclose(phi1_int, 1e-16), 0.0,
            np.where(np.isclose(phi1_int, 1-1e-16), 1.0, phi1_int))
    
    phi2_int_ = np.where(np.isclose(phi2_int, 1e-16), 0.0,
            np.where(np.isclose(phi2_int, 1-1e-16), 1.0, phi2_int))
    
    phi1_b_ = np.linspace(1e-15,1- 1e-15,knots)
    phi2_b_ = 1 - phi1_b_

    phi1_all = np.concatenate([phi1_int_, phi1_b_])
    phi2_all= np.concatenate([phi2_int_, phi2_b_])
    dfdc_all = np.concatenate([dfdc_int, dfdc_b])

    spline = CloughTocher2DInterpolator(list(zip(phi1_all,phi2_all)),dfdc_all)

    if return_knots:
        return spline, phi1_all.size
    else: 
        return spline





