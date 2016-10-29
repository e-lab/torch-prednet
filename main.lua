--Sangpil Kim, Eugenio Culurciello
-- with help from Alfredo Canziani and Abhishek Chaurasia
-- August - September 2016
-- PredNet in Torch8 - from: https://arxiv.org/abs/1605.08104
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

--Set up visual option if needed
if not opt.visOnly then
   os.execute('mkdir '..opt.savedir)
   torch.save(paths.concat(opt.savedir,'opt.t7'),opt)
   print(opt)
else
   local tmp = torch.load(paths.concat(opt.model,'opt.t7'))
   opt.channel = tmp.channel
   opt.nlayers = tmp.nlayers
   opt.useGPU  = tmp.useGPU
   opt.batch   = tmp.batch
   print(opt)
end

print('Using GPU?', opt.useGPU)
print('GPU id?', opt.gpuId)
print('Batch size?', opt.batch)
print('How many layers?' ,opt.nlayers)
print('Keep mode?' ,opt.modelKeep)

--Call files
local U = require 'misc/util'
local loader = require 'misc/data'
local atari  = require 'misc/atari'
local M = require 'models/model'
local Tr= require 'train'
local Te= require 'test'
local Sh= require 'show'

--Setup env
local util  = U(opt)
local gm    = M(opt)
local tr    = Tr(opt)

local te, sh, nets
if opt.visOnly then
   sh = Sh(opt)
else
   if opt.vis or opt.visOnly then sh = Sh(opt) end
   if not opt.trainOnly then te = Te(opt) end
end
--Get model
nets = gm:getModel()

--Main function
local function main()
   --Main loop
   for epoch = 1 , opt.maxEpochs do
      if not opt.visOnly then
         tr:train(util, epoch, nets)
         if not opt.trainOnly then te:test(util, epoch, nets) end
         if opt.vis then sh:show(util) end
      else
         sh:show(util)
      end
      collectgarbage()
   end
end
main()
