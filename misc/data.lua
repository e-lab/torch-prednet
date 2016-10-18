-- dataset and data code inspired by: https://github.com/viorik/ConvLSTM
-- uses moving MNIST numbers to create a video animation and fixed lines to debug
-- SangPil Kim  added batch option
-------------------------------------------------------------------------------
function loadData(big)
  print('Using large dataset?',big)
  local dataFile, datasetSeq
  if big then
    dataFile  = 'dataSets/data-big-train.t7'
    dataFileTest = 'dataSets/data-big-test.t7'
  else
    dataFile  = 'dataSets/data-small-train.t7'
    dataFileTest = 'dataSets/data-small-test.t7'
  end
  return dataFile, dataFileTest
end
function getdataSeq(datafile, big, batch)

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
         seq:resize(batch,nseq,nrows,ncols)
         for j = 1 , batch do
            local i = shuffle[idx]
            seq[j] = data:select(1,i)
         end
         idx = idx + batch
      else
         local i = shuffle[idx]
         seq = data:select(1,i)
         idx = idx + 1
      end
      return seq,i
   end
   if batch > 1 then
      dsample = torch.Tensor(batch,nseq,1,nrows,ncols)
   else
      dsample = torch.Tensor(nseq,1,nrows,ncols)
   end

   setmetatable(datasetSeq, {__index = function(self, index)
                                       local sample,i = self:selectSeq()
                                       dsample:copy(sample)
                                       return {dsample,i}
                                    end})
   return datasetSeq
end
