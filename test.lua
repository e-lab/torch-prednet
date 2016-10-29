-- SangPil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
local class = require 'class'
local Te = class('Te')
require 'paths'
function Te:__init(opt)
   local loader
   if opt.atari then
      loader = require 'misc/atari'
   else
      loader = require 'misc/data'
   end
   print('Loading test data...')
   self.datasetSeq = loader.getdataSeq(paths.concat(opt.dataDir,opt.dataName..'-test.t7'),opt) -- we sample nSeq consecutive frames
   self.testLog = optim.Logger(paths.concat(opt.savedir,'test.log'))
end
function Te:test(util, epoch, protos)
   if util.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   print('==> training model')
   protos.model:evaluate()

   local cerr, ferr, loss = 0, 0, 0

   -- set training iterations and epochs according to dataset size:
   print('Validation epoch #', epoch)

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
      --Get output
      -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
      local output = protos.model:forward(inTableG0)
      local tcerr , tferr , tloss = util:computMatric(targetC, targetF, output)
      -- estimate f and gradients
      -- Calculate Error and sum
      cerr = cerr + tcerr
      ferr = ferr + tferr
      loss = loss + tloss
      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        -- Display
        if util.display then
            local disFlag = 'test'
           util:show(inTableG0[#inTableG0], targetF, targetC, output[1], disFlag)
        end
      end
   end
   --Batch is not divided since it is calcuated already in criterion
   cerr = cerr/iteration/util.batch
   ferr = ferr/iteration/util.batch
   loss = loss/iteration/util.batch
   util:writLog(cerr,ferr,loss,self.testLog)
   print ('Validation completed!')
end

return Te
