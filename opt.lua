-- SangPil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
local o = {}
function o.parse(arg)
   local lapp = require 'pl.lapp'
   local opt = lapp [[
     Command line options:
     --seed                (default 1250)     initial random seed
     --useGPU                                 use GPU in training
     --gpuId               (default 1)        select GPU
     Data parameters:
     --dataDir             (default dataSets)  Dataset dirs
     --dataName            (default data-small) Dataset name
     --atari                                 Use atari
     --model               (default testV)       Model dir

     Training parameters:
     -r,--learningRate       (default 1e-3)  learning rate
     -c,--channel            (default 1)     channel of data
     -d,--learningRateDecay  (default 0)     learning rate decay
     -w,--weightDecay        (default 0)     L2 penalty on the weights
     -m,--momentum           (default 0.9)   momentum parameter
     --maxEpochs             (default 100)   max number of training epochs
     --iteration             (default 0)     like to set own iteration default dataSize
     --batch                 (default 10)    batch size
     --trainOnly                             Train only if true
     --modelKeep                             Long video
     --vis                                   Show hidden states
     --visOnly                               Show only without training
     --visIdx                (default 1)     Pointer for output table index to visualize

     Save options:
     --savedir         (default 'testV') subdirectory to save experiments in
     -s,--save                               save models
     --multySave                             save models respect to saveEpoch
     --saveEpoch             (default 1 )    Save every period epoch
     --savePics                              save output images examples
     --picFreq               (default 10)    if savePics on frequency of save pic

     Model parameters:
     --nlayers               (default 3)     number of layers of MatchNet
     --lstmLayers            (default 1)     number of layers of ConvLSTM
     --inputSizeW            (default 64)    width of each input patch or image
     --inputSizeH            (default 64)    width of each input patch or image
     --nSeq                  (default 20)    input video sequence lenght
     --stride                (default 1)     stride in convolutions
     --padding               (default 1)     padding in convolutions
     --poolsize              (default 2)     maxpooling size

     Display:
     -v, --verbose                           verbose output
     --display                               display stuff
     --disFreq               (default 10)    display and save pic every freq
   ]]
   return opt
end

return o
