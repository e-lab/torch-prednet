-- SangPil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
require 'paths'
local class = require 'class'
local Tr = class('Tr')
function Tr:__init(opt)
   local loader
   if opt.atari then
      loader = require 'misc/atari'
   else
      loader = require 'misc/data'
   end
   print('Loading train data...')
   self.datasetSeq = loader.getdataSeq(paths.concat(opt.dataDir,opt.dataName..'-train.t7'),opt) -- we sample nSeq consecutive frames
   self.trainLog = optim.Logger(paths.concat(opt.savedir,'train.log'))
   self.prevLoss = 1e10
   --Init selfimState
   self.optimState = {
     learningRate = opt.learningRate,
     momentum = opt.momentum,
     learningRateDecay = opt.learningRateDecay,
     weightDecay = opt.weightDecay
   }
end
function Tr:train(util, epoch, protos)
   print('==> training model')
   protos.model:training()
   local w, dE_dw = protos.model:getParameters()
   print('Number of parameters ' .. w:nElement())
   print('Number of grads ' .. dE_dw:nElement())

   local cerr, ferr, loss= 0, 0, 0
   -- set training iterations and epochs according to dataset size:
   print('Training epoch #', epoch)

   local iteartion
   if util.iteration == 0 then
      iteration = math.floor(self.datasetSeq:size()/util.batch)
   else
      iteration = util.iteration
   end
   local output
   for t = 1, iteration do
      xlua.progress(t, iteration)
      -- define eval closure
      local eval_E = function(w)

         protos.model:zeroGradParameters()
         local sample = self.datasetSeq[t]
         local inTableG0, targetC, targetF
         if util.modelKeep and t ~= 1 then
            inTableG0, targetC, targetF = util:prepareDataKeep(sample,output)
         else
            inTableG0, targetC, targetF = util:prepareData(sample)
         end
         --[[
         print('E sum',inTableG0[1]:sum())
         print('C sum',inTableG0[2]:sum())
         print('H sum',inTableG0[3]:sum())
         --]]
         --Get output
         -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
         output = protos.model:forward(inTableG0)
         -- Criterion is embedded
         -- estimate f and gradients
         -- Update Grad input
         local dE_dy
         if self.modelKeep then
            dE_dy = util:prepareDedwKeep(output, targetF)
         else
            dE_dy = util:prepareDedw(output, targetF)
         end
         protos.model:backward(inTableG0,dE_dy)

         -- Display and Save picts
         if math.fmod(t*util.batch, util.disFreq) == 0 then
            local disFlag = 'train'
            util:show(inTableG0[#inTableG0], targetF, targetC, output, disFlag)
            util:saveImg(targetF,output[1],epoch,t, disFlag)
         end
         --Calculate Matric
         -- Calculate Error and sum
         local tcerr , tferr , f = util:computMatric(targetC, targetF, output)
         cerr = cerr + tcerr
         ferr = ferr + tferr
         -- return f and df/dw
         return f, dE_dw
      end
      --Update model
      local _,fs = optim.adam(eval_E, w, self.optimState)
      -- compute statistics / report error
      loss = loss + fs[1]
      --------------------------------------------------------------------
   end
   if self.prevLoss > loss then self.prevLoss = loss end
   -- Save model
   if math.fmod(epoch, util.saveEpoch) == 0 and self.prevLoss == loss then
      --Only save unit model
      protos.clone:evaluate()
      protos.clone:clearState()
      util:saveM(protos.clone, self.optimState, epoch)
   end
   --Average errors
   --Batch is not divided since it is calcuated already in criterion
   cerr = cerr/iteration/util.batch
   ferr = ferr/iteration/util.batch
   loss = loss/iteration/util.batch
   util:writLog(cerr,ferr,loss,self.trainLog)
   print('Learning Rate: ', self.optimState.learningRate)
   print ('Training completed!')
end
return Tr
