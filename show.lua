-- SangPil Kim
-- August - September 2016
-------------------------------------------------------------------------------
local class = require 'class'
local Sh = class('Sh')
require 'paths'
function Sh:__init(opt)
   local loader
   if opt.atari then
      loader = require 'misc/atari'
   else
      loader = require 'misc/data'
   end
   print('Loading show data...')
   self.datasetSeq = loader.getdataSeq(paths.concat(opt.dataDir,opt.dataName..'-train.t7'),opt) -- we sample nSeq consecutive frames
   self.opt = opt
end
function Sh:show(util)
   print  ('Loaded ' .. self.datasetSeq:size() .. ' images')
   print ('load model')
   --model = torch.load(paths.concat(util.model,'model.net'))
   local Ts = require 'models/toSeq'
   local ts = Ts(self.opt)
   model = ts:getModel()
   model:evaluate()
   model:clearState()

   print('Show Start !')
   local iteration
   if util.iteration == 0 then
      iteration = math.floor(self.datasetSeq:size()/util.batch)
   else
      iteration = util.iteration
   end

   for t = 1, iteration do
      xlua.progress(t, iteration)
      local sample = self.datasetSeq[t]
      local inTableG0, targetC, targetF = util:prepareData(sample)
      -- Forward pass
      local output = model:forward(inTableG0)
      --------------------------------------------------------------------
      -- Show states output
      util:show(inTableG0[#inTableG0], targetF, targetC, output, disFlag, util.visIdx)
   end
end

return Sh
