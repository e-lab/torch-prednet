--Sangpil Kim, Eugenio Culurciello
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
local opt = of.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)
print('Using GPU?', opt.useGPU)
print('How many layers?' ,opt.nlayers)

--Call files
local U = require 'misc/util'
local loader = require 'misc/data.lua'
local M = require 'models/model'
local Tr= require 'train'
local Te= require 'test'

local util = U(opt)
local initM = M(opt)
local tr    = Tr(opt)
local te    =Te(opt)
local model = initM:getModel()

local function main()
   print('Loading data...')
   local dataFile, dataFileTest = loader.loadData(opt.dataBig)
   local datasetSeq = loader.getdataSeq(dataFile,opt) -- we sample nSeq consecutive frames
   local testDatasetSeq = loader.getdataSeq(dataFileTest,opt) -- we sample nSeq consecutive frames
   local trainLog = optim.Logger(paths.concat(opt.savedir,'train.log'))
   local testLog = optim.Logger(paths.concat(opt.savedir,'test.log'))
   --Main loop
   for epoch = 1 , opt.maxEpochs do
      tr:train(util, datasetSeq, epoch, trainLog, model)
      te:test(util, testDatasetSeq, epoch, testLog, model)
      collectgarbage()
   end
end

main()
