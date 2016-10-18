-- Sangpil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
function setup(opt)
   --Get model
   print('Initialize model')
   model = getModel()
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
      cutorch.setDevice(opt.GPUID)
      model:cuda()
   end
   --Init optimState
   optimState = {
     learningRate = opt.learningRate,
     momentum = opt.momentum,
     learningRateDecay = opt.learningRateDecay,
     weightDecay = opt.weightDecay
   }
end
function computMatric(targetC, targetF, output)
   local criterion = nn.MSECriterion()
   local cerr = criterion:forward(targetC:squeeze(),output[1]:squeeze())
   local ferr = criterion:forward(targetF:squeeze(),output[1]:squeeze())
   local f = 0
   local numE = #output - 1
   for i = 1 , numE do
      f = f + output[i+1]:sum()
   end
   f = f/numE
   return cerr, ferr, f
end
function writLog(cerr,ferr,loss,logger)
   print(string.format('cerr : %.4f ferr: %.4f loss: %.2f',cerr, ferr, loss))
   logger:add{
      ['cerr'] = cerr,
      ['ferr']  = ferr,
      ['loss'] = loss
   }
end
function shipGPU(table)
   for i,item in pairs(table) do
      table[i] = item:cuda()
   end
end
function prepareDedw(output,targetF)
   local dE_dy = {}
   local criterion = nn.MSECriterion():cuda()
   local dp_dy = criterion:backward(output[1],targetF)
   --table.insert(dE_dy,torch.zeros(output[1]:size()):cuda())
   table.insert(dE_dy,dp_dy)
   for i = 1 , #output - 1 do
      table.insert(dE_dy,output[i+1])
   end
   return dE_dy
end
function prepareData(opt, sample)
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   -- reset initial network state:
   local inTableG0 = {}
   local batch = opt.batch
   for L=1, opt.nlayers do
      if opt.batch > 1 then
         table.insert( inTableG0, torch.zeros(batch,2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)) ) -- E(t-1)
         table.insert( inTableG0, torch.zeros(batch,opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
         table.insert( inTableG0, torch.zeros(batch,opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- H(t-1)
      else
         table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)) ) -- E(t-1)
         table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
         table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- H(t-1)
      end
   end
   -- get input video sequence data:
   local seqTable = {} -- stores the input video sequence
   --sample is the table
   local data = sample[1]
   local nSeq, flag
   if opt.batch > 1 then
      nSeq = data:size(2)
      flag = 2
   else
      nSeq = data:size(1)
      flag = 1
   end
   for i = 1, nSeq do
      table.insert(seqTable, data:select(flag,i)) -- use CPU
   end
   --Ship to GPU
   if opt.useGPU then
      shipGPU(inTableG0)
      shipGPU(seqTable)
   end
   -- prepare table of states and input:
   table.insert(inTableG0, seqTable)
   -- Target
   local targetC, targetF = torch.Tensor(), torch.Tensor()
   if opt.batch == 1 then
      --Extract last sequence to do metric
      targetF:resizeAs(data[nSeq]):copy(data[nSeq])
      targetC:resizeAs(data[nSeq]):copy(data[nSeq-1])
   else
      targetF:resizeAs(data[{{},nSeq,{},{}}]):copy(data[{{},nSeq,{},{}}])
      targetC:resizeAs(data[{{},nSeq-1,{},{}}]):copy(data[{{},nSeq-1,{},{}}])
   end
   if opt.useGPU then
      targetF = targetF:cuda()
      targetC = targetC:cuda()
      data    = data:cuda()
   end
   return inTableG0, targetC, targetF
end
function display(opt, seqTable,targetF,targetC,output, flag)
   if opt.display then
      if flag == 'train' then
        legend = 'Train: t-3, t-2, t-1, Target, Prediction'
      else
        legend = 'Test: t-3, t-2, t-1, Target, Prediction'
      end
      require 'env'
      local pic
      if opt.batch == 1 then
         pic = { seqTable[#seqTable-2]:squeeze(),
                          seqTable[#seqTable-2]:squeeze(),
                          targetC:squeeze(),
                          targetF:squeeze(),
                          output:squeeze() }
      else
         pic = { seqTable[#seqTable-2][1]:squeeze(),
                          seqTable[#seqTable-2][1]:squeeze(),
                          targetC[1]:squeeze(),
                          targetF[1]:squeeze(),
                          output[1]:squeeze() }
      end
      _im1_ = image.display{image=pic, min=0, max=1, win = _im1_, nrow = 7,
                         legend = legend}
   end
end
function savePics(opt,target,output,epoch,t, disFlag)
   --Save pics
   print('Save pics!')
   if disFlag ~= 'train' then disFlag = 'test' end
   if opt.savePics then
      if opt.batch > 1 then
         target = target[1]:squeeze()
         output = output[1]:squeeze()
      end
      image.save(paths.concat(opt.savedir ,'pic_target_'..epoch..'_'..t..'_'..disFlag..'.jpg'), target)
      image.save(paths.concat(opt.savedir ,'pic_output_'..epoch..'_'..t..'_'..disFlag..'.jpg'), output)
   end
end
function save( model, optimState, opt, epoch)
   --Save models
   if opt.save  then
      print('Save models!')
      if opt.multySave then
         torch.save(paths.concat(opt.savedir ,'model_' .. epoch .. '.net'), model)
         torch.save(paths.concat(opt.savedir ,'optimState_' .. epoch .. '.t7'), optimState)
         torch.save(paths.concat(opt.savedir ,'opt' .. epoch .. '.t7'), opt)
      else
         torch.save(paths.concat(opt.savedir ,'model.net'), model)
         torch.save(paths.concat(opt.savedir ,'optimState.t7'), optimState)
         torch.save(paths.concat(opt.savedir ,'opt.t7'), opt)
      end
   end
end
