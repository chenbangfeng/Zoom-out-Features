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











