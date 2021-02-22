import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from aniso.tensor import DiffusionTensor2D
from aniso.derivatives import apply_isotropic_smoothing
from aniso.utils import gridsmooth

if __name__ == "__main__":

    print('Reading image...')
    # Load an image and compute the corresponding diffusion tensor
    image = np.asarray(Image.open('test_images/crabnebula.jpg'))

    print('Computing diffusion tensor...')
    alpha = 50
    dtensor = DiffusionTensor2D.fromimage(image)

    # plot tensor overlay
    plt.imshow(image)
    dtensor.plot_tensor(image, color='green')
    plt.savefig('test_images/structure_tensor.png', dpi=200, bbox_inches='tight', format='png')
    plt.show()

    # Anisotropic smoothing
    iso_image = np.zeros_like(image)
    for i in range(3):
        iso_image[:, :, i] = apply_isotropic_smoothing(image[:, :, i], alpha=alpha)

    aniso_image = dtensor.smooth(image, alpha=alpha)

    # plot comparison
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[1].imshow(iso_image)
    ax[2].imshow(aniso_image)
    titles=['Original', 'Regular smoothing', 'Anisotropic smoothing']

    for axis, title in zip(ax, titles):
        axis.set_title(title)

    plt.savefig('test_images/compare.png', dpi=200, bbox_inches='tight', format='png')
    plt.show()

    # Apply to random vector
    noise = np.random.normal(0, 1, size=image.shape[:2])
    noise = gridsmooth(noise, 2)
    noise /= np.sqrt(np.var(noise))
    textured_noise = dtensor.smooth(noise, alpha=alpha)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(8, 4))
    ax[0].imshow(noise, cmap='RdBu', clim=[-1.5, 1.5])
    ax[1].imshow(textured_noise, cmap='RdBu', clim=[-1.5, 1.5])

    titles=['Random noise', 'Textured noise']
    for axis, title in zip(ax, titles):
        axis.set_title(title)
    plt.savefig('test_images/noise.png', dpi=200, bbox_inches='tight', format='png')

    plt.show()



