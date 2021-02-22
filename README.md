<p align="center">
  <img src="https://github.com/gianmatharu/Aniso/blob/main/test_images/aniso.png?raw=true" alt="Header"/>
</p>

A prototype module to apply anisotropic smoothing to images and explore Dash. 


## Usage 

To generate the example sequence of images, run the following from the command line
```python
>> python main.py
```
The test script loads an image and computes a diffusion tensor. A diffusion tensor is a 2x2 matrix that contains
information about coherent structure in an image. Each pixel has its own diffusion tensor computed using nearby
points. The following example displays local diffusion tensors as ellipses. 
<p align="center">
  <img src="https://github.com/gianmatharu/Aniso/blob/main/test_images/structure_tensor.png?raw=true" alt="Header" width="300" height="300"/>
</p>

We can use diffusion tensors to apply smoothing along directions of structure in an image. Notice how regular smoothing blurs the image everywhere.
<p align="center">
  <img src="https://github.com/gianmatharu/Aniso/blob/main/test_images/compare.png?raw=true" alt="Header" height="300"/>
</p>

We can also use the diffusion tensor to generate texture noise vectors. This can be useful when we want to simulate noise with local correlations.
<p align="center">
  <img src="https://github.com/gianmatharu/Aniso/blob/main/test_images/noise.png?raw=true" alt="Header" height="300"/>
</p>
