-- Sangpil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
local class = require 'class'
local util = class('util')
function util:__init(opt)
   print('init util')
   --Set up optiopn
   for name, value in pairs(opt) do
      self[name] = value
   end
   for i =1 , self.nlayers do
      if i == 1 then
         self.nFilters  = {1} -- number of filters in the encoding/decoding layers
      else
         table.insert(self.nFilters, (i-1)*32)
      end
   end
end
function util:computMatric(targetC, targetF, output)
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
function util:writLog(cerr,ferr,loss,logger)
   print(string.format('cerr : %.4f ferr: %.4f loss: %.2f',cerr, ferr, loss))
   logger:add{
      ['cerr'] = cerr,
      ['ferr']  = ferr,
      ['loss'] = loss
   }
end
function util:shipGPU(table)
   for i,item in pairs(table) do
      table[i] = item:cuda()
   end
end
function util:prepareDedw(output,targetF)
   local dE_dy = {}
   local criterion = nn.MSECriterion()
   if self.useGPU then
      criterion:cuda()
   end
   local dp_dy = criterion:backward(output[1],targetF)
   --table.insert(dE_dy,torch.zeros(output[1]:size()):cuda())
   table.insert(dE_dy,dp_dy)
   for i = 1 , #output - 1 do
      table.insert(dE_dy,output[i+1])
   end
   return dE_dy
end
function util:prepareData(sample)
   if self.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   -- reset initial network state:
   local inTableG0 = {}
   local batch = self.batch
   for L=1, self.nlayers do
      if self.batch > 1 then
         table.insert( inTableG0, torch.zeros(batch,2*self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1)) ) -- E(t-1)
         table.insert( inTableG0, torch.zeros(batch,self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1))) -- C(t-1)
         table.insert( inTableG0, torch.zeros(batch,self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1))) -- H(t-1)
      else
         table.insert( inTableG0, torch.zeros(2*self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1)) ) -- E(t-1)
         table.insert( inTableG0, torch.zeros(self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1))) -- C(t-1)
         table.insert( inTableG0, torch.zeros(self.nFilters[L], self.inputSizeW/2^(L-1), self.inputSizeW/2^(L-1))) -- H(t-1)
      end
   end
   -- get input video sequence data:
   local seqTable = {} -- stores the input video sequence
   --sample is the table
   local data = sample[1]
   local nSeq, flag
   if self.batch > 1 then
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
   if self.useGPU then
      self:shipGPU(inTableG0)
      self:shipGPU(seqTable)
   end
   -- prepare table of states and input:
   table.insert(inTableG0, seqTable)
   -- Target
   local targetC, targetF = torch.Tensor(), torch.Tensor()
   if self.batch == 1 then
      --Extract last sequence to do metric
      targetF:resizeAs(data[nSeq]):copy(data[nSeq])
      targetC:resizeAs(data[nSeq]):copy(data[nSeq-1])
   else
      targetF:resizeAs(data[{{},nSeq,{},{}}]):copy(data[{{},nSeq,{},{}}])
      targetC:resizeAs(data[{{},nSeq-1,{},{}}]):copy(data[{{},nSeq-1,{},{}}])
   end
   if self.useGPU then
      targetF = targetF:cuda()
      targetC = targetC:cuda()
      data    = data:cuda()
   end
   return inTableG0, targetC, targetF
end
function util:show(seqTable,targetF,targetC,output, flag)
   if self.display then
      if flag == 'train' then
        legend = 'Train: t-3, t-2, t-1, Target, Prediction'
      else
        legend = 'Test: t-3, t-2, t-1, Target, Prediction'
      end
      require 'env'
      local pic
      if self.batch == 1 then
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
function util:saveImg(target,output,epoch,t, disFlag)
   --Save pics
   if disFlag ~= 'train' then disFlag = 'test' end
   if self.savePics then
      print('Save pics!')
      if self.batch > 1 then
         target = target[1]:squeeze()
         output = output[1]:squeeze()
      end
      image.save(paths.concat(self.savedir ,'pic_target_'..epoch..'_'..t..'_'..disFlag..'.jpg'), target)
      image.save(paths.concat(self.savedir ,'pic_output_'..epoch..'_'..t..'_'..disFlag..'.jpg'), output)
   end
end
function util:saveM( model, selfimState, epoch)
   --Save models
   if self.save  then
      print('Save models!')
      if self.multySave then
         torch.save(paths.concat(self.savedir ,'model_' .. epoch .. '.net'), model)
         torch.save(paths.concat(self.savedir ,'selfimState_' .. epoch .. '.t7'), selfimState)
         torch.save(paths.concat(self.savedir ,'self' .. epoch .. '.t7'), self)
      else
         torch.save(paths.concat(self.savedir ,'model.net'), model)
         torch.save(paths.concat(self.savedir ,'selfimState.t7'), selfimState)
         torch.save(paths.concat(self.savedir ,'self.t7'), self)
      end
   end
end
function util:printop()
   print('option')
   print(self)
end
return util
