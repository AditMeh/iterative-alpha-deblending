# iterative-alpha-deblending
Testing out [iterative alpha deblending](https://arxiv.org/abs/2305.03486)



## Implementing [RePaint](https://arxiv.org/abs/2201.09865)

Here are a few visualiations of the inpainting process for a batch of images at different levels of $U$. The higher the $U$, the more the masked and unmasked regions are "homogenized". 

The top is the original, the middle is the image with the mask and the bottom is the inpainted image. 

### $U=1$ 

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_1.png?raw=true)

### $U=2$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_2.png?raw=true)

### $U=3$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_3.png?raw=true)

### $U=4$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_4.png?raw=true)

### $U=5$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_5.png?raw=true)

### $U=6$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_6.png?raw=true)

### $U=7$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_7.png?raw=true)

### $U=7$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_8.png?raw=true)

### $U=9$

![U=1](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/test_9.png?raw=true)


## Visualizing the reverse process on MNIST

![alt](https://github.com/AditMeh/iterative-alpha-deblending/blob/main/visualizations/mnist_video.gif?raw=true)