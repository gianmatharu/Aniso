import numpy as np
from scipy.ndimage import correlate1d
from scipy.sparse.linalg import LinearOperator, cg
from scipy.ndimage.filters import laplace
from functools import partial

from aniso.utils import as_ndarrays, check_2d


def compute_derivatives(image):
    """ Compute gradient of 2D image with centered finite difference approximation.

    Parameters
    ----------
    image: array_like
        Input image.

    Returns
    -------
    imx, imy: ndarray, ndim=2
        Gradients in x, y directions respectively.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional!')

    # 2nd order stencil
    stencil = [-0.5, 0, 0.5]

    imx = correlate1d(image, stencil, 1, mode='constant', cval=0.0)
    imy = correlate1d(image, stencil, 0, mode='constant', cval=0.0)

    return imx, imy


def compute_hessian(image):
    """ Compute Hessian components of 2D image using finite difference.

    Parameters
    ----------
    image: array_like
        Input image.

    Returns
    -------
    hxx, hxy, hyy: ndarray, ndim=2
        Gradients in x, y directions respectively.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional!')

    hyy = correlate1d(image, [1, -2, 1], 0, mode='constant', cval=0.0)
    hxx = correlate1d(image, [1, -2, 1], 1, mode='constant', cval=0.0)

    imx = correlate1d(image, [-0.5, 0, 0.5], 1, mode='constant', cval=0.0)
    hxy = correlate1d(imx, [-0.5, 0, 0.5], 0, mode='constant', cval=0.0)

    return hxx, hxy, hyy

# private functions
def _apply_laplacian(nx, ny, alpha, image):
    """ Apply isotropic Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of 2D image
    alpha: float
        Diffusion parameter
    image: ndarray, ndim=1
        Image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of Laplacian
    """
    # reshape
    n = len(image)
    image = image.reshape((ny, nx))

    output = image - alpha * laplace(image)
    output = output.reshape(n)

    return output


def _solve_laplacian(image, alpha=1.0, maxiter='20'):
    """ Use CG to solve a second order differential equation.

    Parameters
    ----------
    image: ndarray, ndim=2
       Input image
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    n = image.size
    (ny, nx) = image.shape

    image = image.reshape(n)
    pmatvec = partial(_apply_laplacian, nx, ny, alpha)
    A = LinearOperator((n, n), matvec=pmatvec)

    output, _ = cg(A, image, maxiter=maxiter)
    output = output.reshape((ny, nx))

    return output


def _apply_directional_laplacian(nx, ny, d11, d12, d22, alpha, image):
    """ Apply the directional Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of image.
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameters
    image: ndarray, ndim=1
        Input image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # reshape to 2D array
    n = image.size
    image = image.reshape((ny, nx))

    # compute first and second derivatives of image
    imx, imy = compute_derivatives(image)

    Dx = d11 * imx + d12 * imy
    Dy = d12 * imx + d22 * imy

    # apply Neumann boundary conditions (normal derivative = 0)
    Dx[:, 0] = 0.0
    Dx[:, -1] = 0.0

    Dy[0, :] = 0.0
    Dy[-1, :] = 0.0

    # compute derivatives of spatially varying structure tensor fields.
    Dxx, _ = compute_derivatives(Dx)
    _, Dyy = compute_derivatives(Dy)

    output = image - alpha * (Dxx + Dyy)
    output = output.reshape(n)

    return output


def _imapply_directional_laplacian(image, d11, d12, d22, alpha):
    """ Apply the directional Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of image.
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameters
    image: ndarray, ndim=1
        Input image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # compute first and second derivatives of image
    imx, imy = compute_derivatives(image)

    Dx = d11 * imx + d12 * imy
    Dy = d12 * imx + d22 * imy

    # apply Neumann boundary conditions (normal derivative = 0)
    Dx[:, 0] = 0.0
    Dx[:, -1] = 0.0

    Dy[0, :] = 0.0
    Dy[-1, :] = 0.0

    # compute derivatives of spatially varying structure tensor fields.
    Dxx, _ = compute_derivatives(Dx)
    _, Dyy = compute_derivatives(Dy)

    output = image - alpha * (Dxx + Dyy)

    return output


def _solve_directional_laplacian(image, d11, d12, d22, alpha=1.0, maxiter=500):
    """ Use CG to solve an anisotropic second order differential equation.

    Parameters
    ----------
    image: ndarray, ndim=2
       Input image
    d11, d12, d22: ndarray, ndim=2def  99 

        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameter
    maxiter: int
        Maximum number of CG iterations

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # get dimensions
    n = image.size
    (ny, nx) = image.shape

    # vectorize for CG
    image = image.reshape(n)

    # partial function for matrix vector product
    pmatvec = partial(_apply_directional_laplacian, nx, ny, d11, d12, d22, alpha)

    # linear operator
    A = LinearOperator((n, n), matvec=pmatvec)

    # solve linear system using CG
    output, _ = cg(A, image, maxiter=maxiter)
    output = output.reshape((ny, nx))

    return output


# smoothing functions
def apply_isotropic_smoothing(image, alpha=1.0):
    """ Smooth an image using Laplacian smoothing.

    Parameters
    ----------
    image: array_like
        Input image.
    alpha: float
        Diffusion parameter.
    -------
    output: ndarray
        Isotropically smoothed image.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional.')

    return _solve_laplacian(image, alpha=alpha)


def apply_anisotropic_smoothing(image, d11, d12, d22, alpha=1.0, maxiter=500):
    """ Apply anisotropic smoothing along coherent directions in image.
        Solves a directional second order PDE that is comparable to an
        anisotropic 2D diffusion equation.

    Parameters
    ----------
    image: array_like
        Input image.
    d11, d12, d22: array_like
        Diffusion tensor coefficients
    alpha: float
        Diffusion parameter.
    maxiter: int
        Max iterations for linear solver 

    Returns
    -------
    output: ndarray
        Anisotropically smoothed image.
    """
    image, d11, d12, d22 = as_ndarrays(image, d11, d12, d22)
    check_2d(image, d11, d12, d22)

    return _solve_directional_laplacian(image, d11, d12, d22, alpha=alpha, maxiter=maxiter)