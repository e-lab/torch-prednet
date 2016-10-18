# PredNet in Torch7

A model of [PredNet](https://coxlab.github.io/prednet/) in [Torch7](http://torch.ch/)

![sample](sample.jpg)

## get started:

Download dataset: moving [MNIST examples](https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa). Small is a 100-sample test, otherwise use the larger ones with 8000 samples. This dataset originated from [Viorica Patraucean and team](http://mi.eng.cam.ac.uk/~vp344/).


## to train PredNet:

Run: ```qlua main.lua -useGPU -dataBig -nlayers 3 -display -save -savePics -batch 10``` to train on GPU on large dataset, visualize and save model and results.


### Notes:

- Memory usage: decrease `nSeq` parameter to use less memory. The model is cloned `nSeq` times and therefore memory usage is proportional to it.
