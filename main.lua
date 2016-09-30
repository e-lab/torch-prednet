-- Eugenio Culurciello, SangPil Kim
-- with help from Alfredo Canziani and Abhishek Chaurasia
-- August - September 2016
-- PredNet in Torch7 - from: https://arxiv.org/abs/1605.08104
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
-- download data from: https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa?dl=0
--
-------------------------------------------------------------------------------

require 'nn'
require 'paths'
require 'torch'
require 'image'
require 'optim'
require 'env'
require 'pl'

lapp = require 'pl.lapp'
opt = lapp [[
  Command line options:
  --savedir         (default './results')  subdirectory to save experiments in
  --seed                (default 1250)     initial random seed
  --useGPU                                 use GPU in training

  Data parameters:
  --dataBig                                use large dataset or reduced one
  
  Training parameters:
  -r,--learningRate       (default 0.001)  learning rate
  -d,--learningRateDecay  (default 0)      learning rate decay
  -w,--weightDecay        (default 0)      L2 penalty on the weights
  -m,--momentum           (default 0.9)    momentum parameter
  --maxEpochs             (default 10)     max number of training epochs
  
  Model parameters:
  --nlayers               (default 2)     number of layers of PredNet
  --lstmLayers            (default 1)     number of layers of ConvLSTM
  --inputSizeW            (default 64)    width of each input patch or image
  --inputSizeH            (default 64)    width of each input patch or image
  --nSeq                  (default 19)    input video sequence lenght
  --stride                (default 1)     stride in convolutions
  --padding               (default 1)     padding in convolutions
  --poolsize              (default 2)     maxpooling size

  Display and save parameters:
  -v, --verbose                           verbose output
  --display                               display stuff
  -s,--save                               save models
  --savePics                              save output images examples
]]

opt.nFilters = {1,32,64,128} -- number of filters in the encoding/decoding layers

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

print('Using GPU?', opt.useGPU)

local function main()
  local w, dE_dw

  paths.dofile('data.lua')
  paths.dofile('model.lua')
  -- print('This is the model:', {model})

  -- send everything to GPU if desired
  if opt.useGPU then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(1)
    model:cuda()
    criterion:cuda()
  end

  print('Using large dataset?', opt.dataBig)
  local dataFile, datasetSeq
  if opt.dataBig then
    dataFile  = 'data-big-train.t7'
    dataFileTest = 'data-big-test.t7'
  else
    dataFile  = 'data-small-train.t7'
    dataFileTest = 'data-small-test.t7'
  end

  print('Loading training data...')
  datasetSeq = getdataSeq(dataFile, opt.dataBig) -- get moving MNIST video data
  print  ('Loaded ' .. datasetSeq:size() .. ' images')

  print('==> training model')
  w, dE_dw = model:getParameters()
  print('Number of parameters ' .. w:nElement())
  print('Number of grads ' .. dE_dw:nElement())

  local err = 0
  local epoch = 1
 
  local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
  }
  
  model:training()

  -- set training iterations and epochs according to dataset size:
  opt.dataEpoch = datasetSeq:size() 
  opt.maxIter = opt.dataEpoch * opt.maxEpochs

  -- train:
  for t = 1, opt.maxIter do

    -- define eval closure
    local eval_E = function(w)
      local f = 0
 
      model:zeroGradParameters()

      -- reset initial network state:
      local inTableG0 = {}
      for L=1, opt.nlayers do
        if opt.useGPU then
          table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- E(t-1)
          table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- C(t-1)
          table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- H(t-1)
        else
          table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)) ) -- E(t-1)
          table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
          table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- H(t-1)
        end
      end

      -- get input video sequence data:
      seqTable = {} -- stores the input video sequence
      target = torch.Tensor()
      sample = datasetSeq[t]
      data = sample[1]
      for i = 1, data:size(1)-1 do
        if opt.useGPU then 
          table.insert(seqTable, data[i]:cuda())
        else
          table.insert(seqTable, data[i]) -- use CPU
        end 
      end
      target:resizeAs(data[1]):copy(data[data:size(1)])
      if opt.useGPU then target = target:cuda() end
      
      -- prepare table of states and input:
      table.insert(inTableG0, seqTable)
      
      -- estimate f and gradients
      output = model:forward(inTableG0)
      f = f + criterion:forward(output, target)
      local dE_dy = criterion:backward(output, target)
      model:backward(inTableG0,dE_dy)
      dE_dw:add(opt.weightDecay, w)

      -- return f and df/dX
      return f, dE_dw
    end
   
    if math.fmod(t, opt.dataEpoch) == 0 then
      epoch = epoch + 1
      print('Training epoch #', epoch)
      opt.learningRate = opt.learningRate * 1/2
      optimState.learningRate = opt.learningRate  
    end
    
    _,fs = optim.adam(eval_E, w, optimState)

    err = err + fs[1]

    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t, opt.nSeq) == 1 then
      print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..opt.learningRate )
      
      err = 0
      
      local pic = { seqTable[#seqTable-3]:squeeze(),
                    seqTable[#seqTable-2]:squeeze(),
                    seqTable[#seqTable-1]:squeeze(),
                    seqTable[#seqTable]:squeeze(),
                    target:squeeze(),
                    output:squeeze() }
      if opt.display then
        _im1_ = image.display{image=pic, min=0, max=1, win = _im1_, nrow = 7, 
                            legend = 't-3, t-2, t-1, t, Target, Prediction'}
      end
    end

    if opt.savePics and math.fmod(t, opt.dataEpoch) == 1 and t>1 then
      image.save(opt.savedir ..'/pic_target_'..t..'.jpg', target)
      image.save(opt.savedir ..'/pic_output_'..t..'.jpg', output)
    end

    if opt.save and math.fmod(t, opt.dataEpoch) == 1 and t>1 then
      torch.save(opt.savedir .. '/model_' .. t .. '.net', model)
      torch.save(opt.savedir .. '/optimState_' .. t .. '.t7', optimState)
    end
  
  end
  print ('Training completed!')
  collectgarbage()
end

main()
