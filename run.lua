--------------Load required modules and functions 
dofile('load_modules.lua') 
dofile('models.lua') 
dofile('PascalClassNames.lua')
dofile('Utilslua')
print('Data, class names, and modules loaded')
torch.setdefaulttensortype('torch.FloatTensor')

-------------Set filepaths, parameters and hyper-parameters
trainDataPath = '/home-nfs/jmichaux/project_4/data/trainData.mat'
trainGTPath = '/home-nfs/jmichaux/project_4/data/trainGT.mat'
valDataPath = '/home-nfs/jmichaux/project_4/data/valData.mat'
valGTPath = '/home-nfs/jmichaux/project_4/data/valGT.mat'



params = {
		traintype = 1,
		maxEpochs = 3,
		numClasses = 21,
		dataType = 'cuda',
		batchSize = 1,
		learningRate = .00001,
		weightDecay = 0.001,
		momentum = 0.9,
		save = true
	}

numClasses = params.numClasses
maxEpochs = params.maxEpochs
dataType = params.dataType
batchSize = params.batchSize



----------------load pre-processing module
prepNet = build_prepNet()
prepNet:evaluate()



----------------Load training model
if params.traintype == -1 then
	model = build_segNet()
	modelfilename = 'model_2_epochs.t7'

elseif params.traintype == 0 then
	model = torch.load('model_2_epochs.t7')    
	modelfilename = 'model_1_epoch_FIM.t7'	

else
	old_model = torch.load('model_FIm.t7')
	old_model:cuda()
	model = build_segNet3()
	model:cuda()
	--add weights from old model
	model:get(1).weight[{{},{},{2},{2}}] = old_model:get(1).weight[{{},{},{1}}]
	model:get(3).weight[{{},{},{2},{2}}] = old_model:get(3).weight[{{},{},{1}}]
	model:get(5).weight[{{},{},{2},{2}}] = old_model:get(5).weight[{{},{},{1}}]
	modelfilename = 'model_3x3.t7'
end

criterion = cudnn.SpatialCrossEntropyCriterion()

if params.dataType == 'cuda' then
	model:cuda()
	criterion:cuda()
end

print('model selected')

---------------load training Data
trainData = {
	inputs = load_image_names(trainDataPath),
	targets = load_image_names(trainGTPath)
}
	
trainData['size'] = #trainData.inputs

--load validation data
valData = {
	inputs = load_image_names(valDataPath),
	targets = load_image_names(valGTPath)
}	
	valData['size'] = #valData.inputs--load data


----------------Begin Training
dofile('Training.lua')
for j = 1, maxEpochs do
    train()
	--validate
	model:evaluate()
	val_confusion = validate_model()
end

---------------Save the model
if params.save == true then
	torch.save(modelfilename,model)
	confusionFile = '/home-nfs/jmichaux/Code/valConfusion_3x3.t7'
	torch.save(confusionFile,val_confusion)
end











--cmd = torch.CmdLine()
--cmd:option('-numClasses', 20,'enter number of classes' )
--cmd:option('-classID', 5, 'enter max number of epochs')
--cmd:option('-maxEpochs', 10, 'enter max number of epochs')
--cmd:option('-classWeight', -1, 'enter class weight')
--cmd:option('-backgroundWeight', 1, 'enter background wieght')
--cmd:option('-learningRate', 1e-4, 'enter learning rate')
--cmd:option('-traintype,-1','are you pre-training or training')
--cmd:option('-batchSize', 30, 'enter batch size')
--cmd:option('-initialize',1,'initialize hidden unit')
--cmd:option('-weightDecay', 1e-2,'enter weight decay')
--cmd:option('-momentum',0.95,'enter momentum')
--cmd:option('-dataType', 'cuda', 'enter something')
--cmd:option('-BatchNorm', 0, 'do you want to try batch normalization')
--cmd:option('-saveflag', false, 'enter saveflag')
--cmd:option('-save', 'results2', 'enter something') --name of the directory


--params = cmd:parse(arg)
--params = {}; params.save = 'problem_2_2a';params.saveflag = true;





--Saving a bunch of stuff
--keptModel = model
--if params.saveflag==true then
--	dofile('savePascal.lua')
--	if params.save == 'problem_2_1' then
--		keptModel:evaluate()
--		classValidation = keptModel:forward(valData.data:cuda())  -- run validation 
--		for n = 1,numClasses do
--		savePascal('val',classValidation[{{},n}], n ,params.save)
--		end
--	end
--
--	if params.save == 'problem_2_2a' or 'problem_2_2b' then
--		keptModel:evaluate()
--		val_predictions = torch.Tensor(valData:size(),numClasses):zero():cuda()
--		for z = 1,valData:size(),opt.batchSize  do
--            counts = 0
--            val_inputs = torch.Tensor(math.min(z+opt.batchSize,valData:size()+1)-z,3,227,227):zero()
--            for k = z,math.min(z+opt.batchSize-1,valData:size()) do
--            	counts = counts+1
--                local im_temp_1 = image.load(valData.imlist[k]) --This isn't good because we will write over imlist
--                local im_temp_2 = image.scale(im_temp_1,227,227,'bilinear')*255
--                local im_temp_3 = im_temp_2:index(1,torch.LongTensor{3,2,1}):float()
--                local im_temp_4 = im_temp_3 - mean_im
--                val_inputs[{counts,{},{},{}}] = im_temp_4
--             end
--             local output = model:forward(val_inputs:cuda())
--             val_predictions[{{z,z+val_inputs:size(1) - 1},{}}] = output
--        end		
--		for n = 1,numClasses do
--			savePascal('val',val_predictions[{{},n}], n ,params.save)
--		end
--	        
--         test_predictions = torch.Tensor(testData:size(),numClasses):zero():cuda()
--         for z = 1,testData:size(),opt.batchSize  do
--             counts = 0
--             test_inputs = torch.Tensor(math.min(z+opt.batchSize,testData:size()+1)-z,3,227,227):zero()
--             for k = z,math.min(z+opt.batchSize-1,testData:size()) do
--             	counts = counts+1
--                local im_temp_1 = image.load(testData.imlist[k]) --This isn't good because we will write over imlist
--                local im_temp_2 = image.scale(im_temp_1,227,227,'bilinear')*255
--                local im_temp_3 = im_temp_2:index(1,torch.LongTensor{3,2,1}):float()
--                local im_temp_4 = im_temp_3 - mean_im
--                test_inputs[{counts,{},{},{}}] = im_temp_4
--             end
--             local output = model:forward(test_inputs:cuda())
--             test_predictions[{{z,z+test_inputs:size(1) - 1},{}}] = output
--         end
--         for n = 1,numClasses do
--             savePascal('test',test_predictions[{{},n}], n ,params.save)
--		 end
--
--
--	end
--end
--
----filename = paths.concat(params.save,'meanIOUs.dat')
----	torch.save(filename,epochIOU)
----end
--
--
--
----torch.save(filename,keptModel)
--	--	classTest = keptModel:forward(testData.data:cuda())
--   --classValidation = torch.exp(keptModel:forward(valData.data)):select(2,2)
--   --  --classTest = torch.exp(keptModel:forward(testData.data)):select(2,2)
--   --   
--   --      --local val_submission_name = string.format('%s%s%s','comp2_cls_val_',classNames[classID],'.txt')
--   --        --local valfile = paths.concat(params.save,val_submission_name)
--   --           --os.execute('mkdir -p ' .. sys.dirname(valfile))
--   --            
