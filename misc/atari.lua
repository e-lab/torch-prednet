-- dataset and data code inspired by: https://github.com/viorik/ConvLSTM
-- uses moving MNIST numbers to create a video animation and fixed lines to debug
-- SangPil Kim  added batch option
-------------------------------------------------------------------------------
local loader = {}
require 'image'
function loader.loadData()
  print('Using large dataset? atari')
  local dataFile, datasetSeq
  dataFile  = 'dataSets/atari/500_20/frameTensor.t7'
  return dataFile, dataFileTest
end
function loader.getdataSeq(datafile, opt)
   local batch = opt.batch
   local data
   data = torch.load(datafile) -- if dataset in binary format

   local datasetSeq ={}
   data = data:float()/255.0
   local nsamples = data:size(1)
   local nseq  = data:size(2)
   local nrows = data:size(4)
   local ncols = data:size(5)
   print ('Dataset size: '..nsamples ..' '..nseq..' '..nrows..' '..ncols)
   function datasetSeq:size()
      return nsamples
   end
   function converter(tmp)
      local pack = torch.Tensor()
      pack = pack:resize(20,3,64,64)
      for t = 1, nseq do
         pack[t] = image.scale(tmp[t],64,64,'bilinear')
      end
      return pack
   end

   local idx = 1
   local shuffle = torch.randperm(nsamples)
   function datasetSeq:selectSeq()
      if idx>nsamples then
        shuffle = torch.randperm(nsamples)
        idx = 1
        print ('data: Shuffle the data')
      end
      local seq = torch.Tensor()
      if batch > 1 then
         seq:resize(batch,nseq,3,64,64)
         for j = 1 , batch do
            local i = shuffle[idx]
            local tmp = data:select(1,i)
            seq[j] = converter(tmp)
         end
         idx = idx + batch
      else
         local i = shuffle[idx]
         local tmp = data:select(1,i)
         seq = converter(tmp)
         idx = idx + 1
      end
      return seq,i
   end
   if batch > 1 then
      dsample = torch.Tensor(batch,nseq,3,64,64)
   else
      dsample = torch.Tensor(nseq,3,64,64)
   end

   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample,i = self:selectSeq()
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end
return loader
