import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CloughTocher2DInterpolator


class Pchip2D:
    def __init__(self, x, y, Z):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        # build one 1D PCHIP in x for each fixed y[j]
        self._x_splines = [PchipInterpolator(self.x, Z[:, j])
                           for j in range(len(self.y))]

    def __call__(self, xq, yq):
        # 1) evaluate each row‐spline at xq → W_j
        W = np.array([s(xq) for s in self._x_splines])
        # 2) build PCHIP in y on W_j, evaluate at yq
        return PchipInterpolator(self.y, W)(yq)

    def partial_x(self, xq, yq):
        # 1) get ∂/∂x of each row‐spline at xq → dW_j
        dW = np.array([s.derivative()(xq) for s in self._x_splines])
        # 2) interp those slopes in y, eval at yq
        return PchipInterpolator(self.y, dW)(yq)

    def partial_y(self, xq, yq):
        # 1) get W_j = each row‐spline at xq
        W = np.array([s(xq) for s in self._x_splines])
        # 2) build PCHIP in y on W, take its derivative at yq
        y_spline = PchipInterpolator(self.y, W)
        return y_spline.derivative()(yq)
    

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

    # build and return the tensor‐product PCHIP
    spline2d = Pchip2D(phi1_knots, phi2_knots, Z)
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

    # build and return the tensor‐product PCHIP
    spline2d = Pchip2D(phi1_knots, phi2_knots, Z)
    return spline2d