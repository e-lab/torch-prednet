-- SangPil Kim, Eugenio Culurciello
-- August - September 2016
-------------------------------------------------------------------------------
function test(opt,datasetSeq,epoch,testLog)
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   print('==> training model')
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   model:evaluate()

   local cerr, ferr, loss = 0, 0, 0

   -- set training iterations and epochs according to dataset size:
   print('Validation epoch #', epoch)

   local iteration
   if opt.iteration == 0 then
      iteration = math.floor(datasetSeq:size()/opt.batch)
   else
      iteration = opt.iteration
   end
   for t = 1, iteration do
      xlua.progress(t, iteration)
      local sample = datasetSeq[t]
      local inTableG0, targetC, targetF = prepareData(opt,sample)
      --Get output
      -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
      local output = model:forward(inTableG0)
      local tcerr , tferr , tloss = computMatric(targetC, targetF, output)
      -- estimate f and gradients
      -- Calculate Error and sum
      cerr = cerr + tcerr
      ferr = ferr + tferr
      loss = loss + tloss
      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        -- Display
        if opt.display then
            local disFlag = 'test'
           display(opt, inTableG0[#inTableG0], targetF, targetC, output[1], disFlag)
        end
      end
   end
   --Batch is not divided since it is calcuated already in criterion
   cerr = cerr/iteration/opt.batch
   ferr = ferr/iteration/opt.batch
   loss = loss/iteration/opt.batch
   writLog(cerr,ferr,loss,testLog)
   print ('Validation completed!')
end
