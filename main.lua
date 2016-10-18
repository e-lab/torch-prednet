--Eugenio Culurciello, Sangpil Kim
-- with help from Alfredo Canziani and Abhishek Chaurasia
-- August - September 2016
-- PredNet in Torch7 - from: https://arxiv.org/abs/1605.08104
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
-- download data from: https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa?dl=1
--
-------------------------------------------------------------------------------

require 'nn'
require 'paths'
require 'torch'
require 'image'
require 'optim'
require 'xlua'
require 'pl'
local of = require 'opt'
opt = of.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
print('Using GPU?', opt.useGPU)
print('How many layers?' ,opt.nlayers)

--Call files
paths.dofile('misc/util.lua')
paths.dofile('misc/data.lua')
paths.dofile('models/model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

local function main()
   print('Loading data...')
   local dataFile, dataFileTest = loadData(opt.dataBig)
   local datasetSeq = getdataSeq(dataFile, opt.dataBig, opt.batch) -- we sample nSeq consecutive frames
   local testDatasetSeq = getdataSeq(dataFileTest, opt.dataBig, opt.batch) -- we sample nSeq consecutive frames
   setup(opt)
   trainLog = optim.Logger(paths.concat(opt.savedir,'train.log'))
   testLog = optim.Logger(paths.concat(opt.savedir,'test.log'))
   --Main loop
   for epoch = 1 , opt.maxEpochs do
      train(opt, datasetSeq, epoch, trainLog)
      test(opt, testDatasetSeq, epoch, testLog)
      collectgarbage()
   end
end

main()
