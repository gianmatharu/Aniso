import numpy as np
import skimage.feature as ski
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from aniso.derivatives import apply_anisotropic_smoothing
from aniso.utils import as_ndarrays, check_2d, eigsorted


# class definitions
class StructureTensor2D(object):
    """ 2D structure tensor class. 
        Container for a structure tensor of a 2D image. 

        A structure tensor for image I at pixel (x, y) is 
        defined as grad(I) * grad (I).T (where * denotes)
        the outer product.

        | Ixx Ixy | 
        | Ixy Iyy |
    """
    def __init__(self, t11, t12, t22):
        """ 
        t11, t12, t22 should be structure tensors as 
        output by skimage.feature.structure_tensor
        """
        # check input and sizes
        self.check(t11, t12, t22)

        # Initialize structure tensor
        self.t11 = t11
        self.t12 = t12
        self.t22 = t22

        # t11, t12, t22 should have size nx, ny
        self.shape = t11.shape

    def check(self, t11, t12, t22):
        # verify input
        t11, t12, t22 = as_ndarrays(t11, t12, t22)
        check_2d(t11, t12, t22)

    # alternate constructor methods.
    @classmethod
    def fromimage(cls, image, sigma=1.0):
        """ Construct a structure tensor from an input image.

        Parameters
        ----------
        image: array_like
            Input image. 
        sigma: float, optional.
                standard deviation of Gaussian kernel. 
                (See skimage.feature.structure_tensor)

        Returns
        -------
        output: StructureTensor2D
            Instance of structure tensor.
        """

        image = np.asarray(image)
        if not (image.ndim == 2 or image.ndim == 3):
            raise ValueError('Image must be 2 or 3-dimensional!')

        image = image / abs(image.max())

        if image.ndim == 3:
            t11, t12, t22 = ski.structure_tensor(image[:, :, 0], sigma=sigma, mode='constant', cval=0.0)
        else:
            t11, t12, t22 = ski.structure_tensor(image, sigma=sigma, mode='constant', cval=0.0)

        return cls(t11, t12, t22)

    @classmethod
    def isotropic(cls, nx, ny):
        """  Sets an isotropic, spatially-invariant structure tensor field.

        Parameters
        ----------
        nx, ny: int, required.
            dimensions of 2D field.

        Returns
        -------
        output: StructureTensor2D
            Instance of structure tensor.
        """
        return cls(np.ones((ny, nx)), np.zeros((ny, nx)), np.ones((ny, nx)))

    @property
    def structure_tensor(self):
        """ Return structure tensor as a 3-tuple
            in order [Ixx, Ixy, Iyy]
        """
        return self.t11, self.t12, self.t22

    def get_tensor_at(self, ix, iy):
        return self.t11[iy, ix], self.t12[iy, ix], self.t22[iy, ix]

    def plot_tensor(self, image, ax=None, interval=0.05, color='g'):

        if ax is None:
            ax = plt.gca()

        image = np.asarray(image)
        # if image.shape != self.shape:
        #     raise ValueError('Image does not match structure tensor dimensions.')

        ax.imshow(image)

        ny, nx = self.shape        
        dh = int(max(self.shape) * interval)

        for i in range(0, ny, dh):
            for j in range(0, nx, dh):
                D = [[self.t11[i][j], self.t12[i][j]], [self.t12[i][j], self.t22[i][j]]]
                vals, vecs = eigsorted(D)
                ells = [_draw_eig_ellipse([j ,i], vals, vecs, scale=20.0, fill=False, color=color)]

                for e in ells:
                    ax.add_artist(e)

        return ax


class DiffusionTensor2D(StructureTensor2D):

    def __init__(self, t11, t12, t22):
        super().__init__(t11, t12, t22)

    @classmethod
    def fromimage(cls, image, sigma=10.0, weighted=False):
        """ Construct a structure tensor from an input image.

        Parameters
        ----------
        image: array_like
            Input image.
        sigma: float, optional.
            standard deviation of Gaussian kernel. 
            (See skimage.feature.structure_tensor)
        weighted: boolean, optional
            Defines how eigenvalues of diffusion tensor are 
            computed. weighted=False does not produce isotropic
            diffusion tensors. 

        Returns
        -------
        output: StructureTensor2D
            Instance of structure tensor.
        """
        
        image = np.asarray(image)

        if not (image.ndim == 2 or image.ndim == 3):
            raise ValueError('Image must be 2 or 3-dimensional!')
        
        if image.ndim == 3: 
            t11, t12, t22 = cls.diffusion_tensor(image[:, :, 0], sigma=sigma, weighted=weighted)
        else:
            t11, t12, t22 = cls.diffusion_tensor(image, sigma=sigma, weighted=weighted)

        return cls(t11, t12, t22)

    @staticmethod
    def diffusion_tensor(image, sigma=10.0, weighted=False):
        """ Compute spatially variant diffusion tensor.

        For a given image I, the structure tensor is given by

        | Ixx Ixy |
        | Ixy Iyy |

        with eigenvectors u and v. Eigenvectors v correspond to the smaller
        eigenvalue and point in directions of maximum coherence.
        The diffusion tensor D is given by, D = v * transpose(v).
        This tensor is designed for anisotropic smoothing.
        Local D tensors are singular.

        Parameters
        ----------
        image: array_like
            Input image.
        sigma: float, optional
            Standard deviation of Gaussian kernel.
            (See skimage.feature.structure_tensor)
        weighted: boolean, optional
            Defines how eigenvalues of diffusion tensor are 
            computed. weighted=False does not produce isotropic
            diffusion tensors. 
        Returns
        -------
        d11, d12, d22: ndarray
            Independent components of diffusion tensor.
        """

        image = np.asarray(image)

        if image.ndim != 2:
            image = np.asarray(image)

        # normalize image
        image = image / abs(image.max())
        (ny, nx) = image.shape

        # # Initialize diffusion tensor components
        d11 = np.zeros((ny, nx))
        d12 = np.zeros((ny, nx))
        d22 = np.zeros((ny, nx))

        # Compute gradient of scalar field using derivative filter
        Ixx, Ixy, Iyy = ski.structure_tensor(image, sigma=sigma, mode='nearest', cval=0.0)

        for i in range(ny):
            for j in range(nx):
                A = [[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]]
                vals, vecs = eigsorted(A)

                alpha = 0.01
                if weighted:
                    if vals[0] == vals[1]:
                        lam1 = 0.5
                        lam2 = 0.5
                    else:
                        linearity = vals[0] - vals[1] / vals[0]
                        lam1 = alpha
                        lam2 = alpha + (1-alpha)*np.exp(-1./linearity**2)

                        lam_sum = lam1 + lam2

                        lam1 /= lam_sum
                        lam2 /= lam_sum
                else: 
                    lam1 = 0.05
                    lam2 = 0.95

                # eigen-decomposition of diffusion tensor
                D = lam1 * np.outer(vecs[:, 0], vecs[:, 0]) + \
                    lam2 * np.outer(vecs[:, -1], vecs[:, -1])

                # Assign components
                d11[i][j] = D[0][0]
                d12[i][j] = D[0][1]
                d22[i][j] = D[1][1]

        return d11, d12, d22


    def smooth(self, image, alpha=10, maxiter=500):
        """ Compute spatially variant diffusion tensor.

        For a given image I, the structure tensor is given by

        | Ixx Ixy |
        | Ixy Iyy |

        with eigenvectors u and v. Eigenvectors v correspond to the smaller
        eigenvalue and point in directions of maximum coherence.
        The diffusion tensor D is given by, D = v * transpose(v).
        This tensor is designed for anisotropic smoothing.
        Local D tensors are singular.

        Parameters
        ----------
        image: array_like
            Input image.
        sigma: float, optional
            Standard deviation of Gaussian kernel.
            (See skimage.feature.structure_tensor)

        Returns
        -------
        smooth_image: ndarray
            Independent components of diffusion tensor.
        """
        if image.ndim == 3:
            smooth_image = np.zeros_like(image)

            # loop over RGB channels
            for i in range(3):
                smooth_image[:, :, i] = apply_anisotropic_smoothing(image[:, :, i], self.t11, self.t12, self.t22, alpha=alpha, maxiter=maxiter)
        else:

            smooth_image = apply_anisotropic_smoothing(image, self.t11, self.t12, self.t22, alpha=alpha, maxiter=maxiter)
        return smooth_image


# plotting utilities
def _draw_eig_ellipse(pos, vals, vecs, scale, **kwargs):
    """ Draw ellipses from eigenvec/val pair
    """

    # Generate ellipse
    eps = 1e-6
    isotropic = 1 / (np.sqrt(vals[0]) + eps)
    linearity = 1 / (np.sqrt(vals[1]) + eps)

    escale = linearity + isotropic

    linearity /= escale
    isotropic /= escale

    width = 0.9 * scale * linearity
    height = 0.9 * scale * isotropic

    theta = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    ellip = Ellipse(xy=pos, width=height, height=width, angle=theta, **kwargs)

    return ellip

