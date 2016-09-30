# PredNet training

A model of [PredNet](https://coxlab.github.io/prednet/) in Torch7

To run:

1: Download dataset: moving [MNIST examples](https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa). Small is a 100-sample test, otherwise use the larger ones with 8000 samples. This dataset originated from [Viorica Patraucean and team](http://mi.eng.cam.ac.uk/~vp344/).


## to train MatchNet:

3: run: ```qlua main.lua -nlayers 3 -display -save -savePics``` to train with 2 layers (1-4 layer possible), visualize and save model and results

GPU supported by ```-useGPU``` option.