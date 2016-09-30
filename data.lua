-- dataset and data code inspired by: https://github.com/viorik/ConvLSTM
-- uses moving MNIST numbers to create a video animation and fixed lines to debug
-------------------------------------------------------------------------------

function getdataSeq(datafile, big)
   
   local data 
   data = torch.load(datafile) -- dataset in binary format
   
   local datasetSeq ={}
   data = data:float()/255.0
   local nsamples = data:size(1)
   local nseq  = data:size(2)
   local nrows = data:size(4)
   local ncols = data:size(5)
   print ('Training data size: '..nsamples ..' '..nseq..' '..nrows..' '..ncols)
   function datasetSeq:size()
      return nsamples
   end

   local idx = 1
   local shuffle = torch.randperm(nsamples)
   function datasetSeq:selectSeq()
      if idx>nsamples then
        shuffle = torch.randperm(nsamples)
        idx = 1
        print ('data: Shuffle the data')
      end
      local i = shuffle[idx]
      local seq = data:select(1,i)
      idx = idx + 1
      return seq,i
   end

   dsample = torch.Tensor(nseq,1,nrows,ncols)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample,i = self:selectSeq()
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end
