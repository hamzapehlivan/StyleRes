### InterfaceGAN

The editing directions are taken from [e4e repository](https://github.com/omertov/encoder4editing/tree/main/editings/interfacegan_directions). 

We will also share the editing directions we found soon.


### GANSpace

The editing code is again modified from e4e repository. The config file is taken from GANSpace [repository](https://github.com/harskish/ganspace/blob/master/notebooks/figure_teaser.ipynb). The last columns of the [config file](/editings/ganspace_pca/ganspace_configs.csv) correspond to editing strenghts.

### StyleClip

Mapping method of StyleClip is modified from the original [repository](https://github.com/orpatashnik/StyleCLIP/tree/main/mapper). The mapper models are also downloaded from the same repo. We modified them so that they include only the mapper network, and not the StyleGAN. The first columns of the [config file](/editings/styleclip_directions/styleclip_mapping_configs.csv) show the possible edits.   

Global directions method is modified from [HyperStyle](https://github.com/yuval-alaluf/hyperstyle/blob/main/editing/styleclip/global_direction.py). In the [examples file](/editings/styleclip_directions/styleclip_global_examples.csv), one can various examples. 

### GradCtrl

GradCtrl directions are calculated using the [official repository](https://github.com/zikuncshelly/GradCtrl/blob/main/manipulate.py). 