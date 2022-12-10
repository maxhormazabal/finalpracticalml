# Legacy code generated in practice classes

# Calculate the normalization parameters for Minimum and Maximum
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Any,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
   
    return convert(NTuple{2, AbstractArray{<:Any,2}}, (mins, maxs))
end

# Normalize the dataset with Min and Max
function normalizeMinMax!(dataset::AbstractArray{<:Any,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Any,2}})
    
    # Get maximum and minimun as vector for use in comparison
    max = vec(normalizationParameters[2])
    min = vec(normalizationParameters[1])
    
    # Normalization formula v'=(v-min)/(max-min). If max equals to min, no normalization is required so constants are used.
    dataset = (dataset.-(max == min ? 0 : normalizationParameters[1]))./(max == min ? 1 : normalizationParameters[2]-normalizationParameters[1])
end

# Normalize the dataset with Min and Max
function normalizeMinMax!(dataset::AbstractArray{<:Any,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

# Normalize the dataset with Min and Max without altering the original dataset
function normalizeMinMax(dataset::AbstractArray{<:Any,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Any,2}}) 
    
    ds = copy(dataset)
    normalizeMinMax!(ds, normalizationParameters)
    return ds
end

# Normalize the dataset with Min and Max without altering the original dataset
function normalizeMinMax(dataset::AbstractArray{<:Any,2})
    ds = copy(dataset)
    
    normalizeMinMax(ds, calculateMinMaxNormalizationParameters(ds))
    return ds
end

# One hot encode the values of the dataset
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})

    numClasses = length(classes);
    @assert(numClasses>1)

    if (numClasses==2)
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        oneHot = BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature.==classes[numClass]);
        end
    end

    return oneHot;
end;

# One hot encode the values of the dataset
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# One hot encode the values of the dataset
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return oneHotEncoding(feature, unique(feature))
end

# Build an ANN model with the number of neurons and functions specified by the user
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann = Chain();
    
    numInputsLayer = numInputs;
    i = 0;
        
    if topology != nothing
        for numOutputsLayer in topology
            i = i + 1;
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]));      
            numInputsLayer = numOutputsLayer; 
        end
    else
        # If no layer, use an identity
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));   
    end
    
    # If is a multiclass problem, use softmax
    if numOutputs > 1
        ann = Chain(ann...,  softmax)
	else
        # If is a binary problem, use identity
		ann = Chain(ann...,  identity)
    end
    
    return ann;
end

# Get the hould out indexes
# N=number of patterns, P=Percentage of patters separated
function holdOut(N::Int, P::Real)
    # Assure that the function have repeatable results for splitting
    Random.seed!(2)

    @assert ((P>=0.) & (P<=1.));
    
    # by using randperm along side with usign random, we guarantee that the values are correct
    indices = randperm(N)
    n_train = Int(round((1 - P)*N))
    return (indices[1:n_train],indices[n_train+1:end])
end

# Get the hould out indexes
function holdOut(N::Int, Pval::Real, Ptest::Real)
    #Check that Pval and Ptest contain values between 0 and 1,
    # as well as the sum of both values should be lower or equal to 1.
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest<=1.));
    
    # First, separate dataset into training+validation and test using previous holdOut method
    dataset_tr_val_tst = holdOut(N, Ptest);

    # Adjust the ratio applied in the second call of the function.
    N_tr_val=N-length(dataset_tr_val_tst[2])
    Pval_tr_val=N*Pval/N_tr_val
    
    # Separate training+validation into training and validation using again previous holdOut method.
    dataset_tr_val = holdOut(N_tr_val, Pval_tr_val);
    
    dataset_tr = dataset_tr_val_tst[1][1:length(dataset_tr_val[1])]
    dataset_val = dataset_tr_val_tst[1][length(dataset_tr_val[1])+1:length(dataset_tr_val_tst[1])]
      
    # Keep in mind that the indexes from the return tuple must be used to obtain the elements from the training+validation set.
    return (dataset_tr,dataset_val,dataset_tr_val_tst[2])
end

# Calculate loss values of the datasets
function calculateLossValues()
    trainingLoss = loss(trainingInputs', trainingTargets');
    validationLoss = loss(validationInputs', validationTargets');
    testLoss = loss(testInputs', testTargets');
    
    push!(trainingLosses, trainingLoss);
    push!(validationLosses, validationLoss);
    push!(testLosses, testLoss);
end

# Create and train an ANN model
function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false) 
    
    # Load inputs and targets
    (trainingInputs, trainingTargets) = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;
    
    # Check with @assert that we have same number of rows (patterns) in training, validation and test sets.
    @assert(size(trainingInputs, 1)==size(trainingTargets, 1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    @assert(size(testInputs, 1)==size(testTargets, 1));
    
    # Check that the number of columns matches in the training and others sets and if the group is not empty
    !isempty(validationInputs) && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(testInputs) && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    
    # Get the amount of neurons the ann should have
    inputneurons = size(trainingDataset[1])[:2]
    outputneurons = size(trainingDataset[2])[:2]
    
    #Builds the neural network
    ann = buildClassANN(inputneurons,topology,outputneurons, transferFunctions=transferFunctions)
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y)
    
    #if no validation, always save last (points to memory)
    bestAnn=ann
    
    numEpoch = 0;
    numEpochsValidation = 0;
    trainingLoss = 100;
    bestValidationLoss = 101;
    validationLoss = 100;
    
    # Load cycle zero loss
    trainingLosses = [loss(trainingInputs', trainingTargets')]
    if (!isempty(validationInputs))
        validationLosses = [loss(validationInputs', validationTargets')]
    else
        validationLosses = []
    end
    if (!isempty(testInputs))
        testLosses = [loss(testInputs', testTargets')]
    else
        testLosses = []
    end
    
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        #It should be noted that the loss values obtained with the ANN with random weights, prior to training, are usually considered as cycle 0
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        numEpoch += 1;
        
        trainingLoss = loss(trainingInputs', trainingTargets');
        push!(trainingLosses, trainingLoss);
        if showText print("Iteration: ", numEpoch ,". Training Loss:", trainingLoss) end
        
        if (!isempty(testInputs))
            testLoss = loss(testInputs', testTargets');
            push!(testLosses, testLoss);
            if showText print(". Test Loss:", testLoss) end
        end
        
        # If validation is active
        if (!isempty(validationInputs))
            validationLoss = loss(validationInputs', validationTargets');
            push!(validationLosses, validationLoss);
            
            # Save the model with the lowest validation loss, checking every cycle if it improves
            if (validationLoss<bestValidationLoss)
                bestValidationLoss=validationLoss;
                bestAnn=deepcopy(ann);
                numEpochsValidation = 0;
            else
                numEpochsValidation += 1;
            end
                if showText println(". Validation Loss:", validationLoss) end
        end       
    end
    if (showText)
        println("Exit: ByMaxEpoch: ", (numEpoch<maxEpochs), " By MinLoss:", (trainingLoss>minLoss), " By EpochVal: ", (numEpochsValidation<maxEpochsVal))
    end
    
    return (bestAnn,[trainingLosses, validationLosses, testLosses]);
end

# Create and train an ANN model
function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)
    
    trainingDataset=(trainingDataset([1], reshape(trainingDataset[2], length(trainingDataset[2]), 1)))
    validationDataset=(trainingDataset([1], reshape(validationDataset[2], length(validationDataset[2]), 1)))
    testDataset=(trainingDataset([1], reshape(testDataset[2], length(testDataset[2]), 1)))

    trainClassANN(topology,trainingDataset,validationDataset,testDataset,transferFunctions,maxEpochs,minLoss,learningRate,maxEpochsVal, showText)
end

# Calculate the confusion matrix and its stadistics
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    @assert(numInstances>0);

    # Calculate the True and False negatives and positives
    TN = sum(.!outputs .& .!targets);
    FN = sum(.!outputs .& targets);
    TP = sum( outputs .& targets);
    FP = sum( outputs .& .!targets);
    confMatrix = [TN FP; FN TP];

    # Metrics derived from the confusión matrix:
    if (TN+TP+FN+FP) > 0
        accuracy = (TN+TP)/(TN+TP+FN+FP)
        errorrate = (FP+FN)/(TN+TP+FN+FP)
    else
        accuracy = 0
        errorrate = 0
    end
    
    if (FN+TP) > 0 # Sensitivity or recall
        if TN!=numInstances
            sensitivity = TP/(FN+TP)  
        else 
            sensitivity = 1
        end
    else
        sensitivity = 0
    end
    
    if (FP+TN) > 0
        if TP!=numInstances
            specificity = TN/(FP+TN)
        else
            specificity = 1
        end
    else
        specificity = 0
    end
        
    if (TP+FP) > 0 # Precision or positive predictive value
        if TN!=numInstances
            PPV = TP/(TP+FP)  
        else 
            PPV = 1
        end
    else
        PPV = 0
    end
            
    if (TN+FN) > 0
        if TP!=numInstances
            NPV = TN/(TN+FN)
        else
            NPV = 1
        end
    else
        NPV = 0
    end
            
    # It is defined as the harmonic mean of precision (positive predictive value) and recall (Sensitivity).
    # (2ab)/(a+b)
    if (sensitivity!=0 || PPV != 0)
        Fscore = (2 * PPV * sensitivity)/(PPV+sensitivity)
    else
        Fscore = 0
    end
    
    return (accuracy, errorrate, sensitivity, specificity, PPV, NPV, Fscore, confMatrix)
end

# Receives two lists of boolean and returns the average of equal values between them
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    tot = length(outputs[outputs .== targets])
    return tot / length(outputs)
end

# Receives two lists of boolean and returns the average of equal values between them
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    if (size(outputs)[:2]>1)
        classComparison = targets .!= outputs 
        incorrectClassifications = any(classComparison, dims=2)
        return 1 - mean(incorrectClassifications)
    else
        accuracy(outputs[1], targets[1])
    end
end

# Returns the accuracy of the values
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    
    #A vector with the boolean classification
    result = Vector{Bool}(undef, length(outputs));
        
    #Compare each value to the threshold
    result.= outputs.>= threshold;
        
    tot = length(result[result .== targets])
    return tot / length(result)
end

# Returns the accuracy of the values
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    
    if (size(outputs)[:2]>1)
        values = classifyOutputs(outputs);
        accuracy(values);
    else
        accuracy(outputs[1]);
    end    
end

# Returns the classification of the values
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 

    if (size(outputs)[:2]>1)
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        
        #A matrix with the boolean classification
        result = falses(size(outputs));
        result[indicesMaxEachInstance] .= true;
        
        return result
    else
        #A vector with the boolean classification
        result = Vector{Bool}(undef, length(outputs));
        
        #Compare each value to the threshold
        result.= outputs.>= threshold;
        
        #Reshape the vector to a matrix
        return reshape(result, length(result), 1)
    end
        
    return result
end

# Returns the values of the metrics adapted to the condition of having more than two classes
# outputs=prediction, targets=actual values
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Check that the number of columns of both matrices is equal 
    @assert(size(outputs, 2)==size(targets, 2));
    # and is different from 2. 
    @assert(size(outputs, 2)!=2);
    @assert(size(targets, 2)!=2);
    
    # In case they have only one column, these columns are taken as vectors and the confusion Matrix previous is called.
    if (size(outputs, 2)==1)
        return confusionMatrix(vec(outputs), vec(targets))
    else
        #Gets the number of classes
        numClasses = size(outputs, 2)
        
        # Reserve memory for the sensitivity, specificity, PPV, NPV and F-score vectors, 
        # with one value per class, initially equal to 0
        stadistics = zeros(numClasses, 5)
        
        # Iterate for each class
        for numClass in 1:numClasses
            outputs_class=outputs[:,[numClass]]
            targets_class=targets[:,[numClass]]
            
            # Count could be used to for this functions
            outputs_patterns=any(outputs_class)
            targets_patterns=any(targets_class)
            
            #if there are patterns in that class
            if (count(outputs_class) > 0 || count(targets_class) > 0)
                res = confusionMatrix(vec(outputs_class), vec(targets_class))
                
                #Assign the result to the corresponding element of the sensitivity, specificity, PPV, NPV and F1 vectors.
                stadistics[numClass,1] = res[3] # Sensitivity
                stadistics[numClass,2] = res[4] # Specificity
                stadistics[numClass,3] = res[5] # PPV
                stadistics[numClass,4] = res[6] # NPV
                stadistics[numClass,5] = res[7] # F-score
            end
        end
        
        # Reserve memory for the confusion matrix.
        confusion_matrix = zeros(numClasses, numClasses)
        
        # Perform a double loop in which booth loops iterate over the classes, to fill all the confusion matrix elements.
        for numRow in 1:numClasses
            targets_class=targets[:,[numRow]][:]
            
            for numCol in 1:numClasses
                outputs_class=outputs[:,[numCol]][:]
               
                # For each class, mark the ones that are true (TP, FP)
                for x in 1:size(outputs_class,1)                    
                    if (outputs_class[x]==true)
                        if (outputs_class[x]==targets_class[x])
                            confusion_matrix[numRow,numCol]=confusion_matrix[numRow,numCol] + 1;
                        end
                    end
                end
            end
        end
        
        # Aggregate the values of sensitivity, specificity, PPV, NPV, and F-score for eachclass into a single value.
        if (weighted) 
            weighted_stadistics = zeros(numClasses, 5)
            # Weighted. In this stratey, the metrics corresponding to each class are averaged, weighting them with the number 
            # of patterns that belong (desired output) to each class. It is therefore suitable when classes are unbalanced.
            for numRow in 1:numClasses
                class_statidistic = vec(stadistics[numRow,:])
                # weigthed = var * numTP / numTot
                weighted_stadistics[numRow, 1] = class_statidistic[1] * confusion_matrix[numRow, numRow] / size(targets,1)
                weighted_stadistics[numRow, 2] = class_statidistic[2] * confusion_matrix[numRow, numRow] / size(targets,2)
                weighted_stadistics[numRow, 3] = class_statidistic[3] * confusion_matrix[numRow, numRow] / size(targets,3)
                weighted_stadistics[numRow, 4] = class_statidistic[4] * confusion_matrix[numRow, numRow] / size(targets,4)
                weighted_stadistics[numRow, 5] = class_statidistic[5] * confusion_matrix[numRow, numRow] / size(targets,5)
            end
            
            sensitivity = mean(weighted_stadistics[:,[1]])
            specificity = mean(weighted_stadistics[:,[2]])
            PPV = mean(weighted_stadistics[:,[3]])
            NPV = mean(weighted_stadistics[:,[4]])
            fscore = mean(weighted_stadistics[:,[5]])
        else
            #Macro. In this strategy, those metrics such as the PPV or the F-score are calculated as the arithmetic mean 
            # of the metrics of each class. As it is an arithmetic average.
            sensitivity = mean(stadistics[:,[1]])
            specificity = mean(stadistics[:,[2]])
            PPV = mean(stadistics[:,[3]])
            NPV = mean(stadistics[:,[4]])
            fscore = mean(stadistics[:,[5]])
        end       
        
        #Finally, calculate the accuracy value with the accuracy function developed in a previous assignment, 
        #and calculate the error rate from this value
        tot_accuracy = accuracy(outputs, targets)
        classComparison = targets .!= outputs 
        incorrectClassifications = any(classComparison, dims=2)
        error_rate = mean(incorrectClassifications)
        
        return (tot_accuracy, error_rate, sensitivity, specificity, PPV, NPV, fscore, confusion_matrix)
    end
end

# Return the confusion matrix of the values
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    values = classifyOutputs(outputs);
    
    confusionMatrix(values, targets, weighted = weighted) 
end

# Return the confusion matrix of the values
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # it is necessary that all the output classes (vector outputs) are included in the desired output classes (vector targets) 
    @assert(all([in(output, unique(targets)) for output in outputs]))
    
    # Use the same list of classes for encoding the same way
    encoded_targets = oneHotEncoding(targets, unique(targets))
    encoded_outputs = oneHotEncoding(outputs, unique(targets))
      
    return confusionMatrix(encoded_outputs, encoded_targets, weighted=weighted)
end

# Prints a generic confusion Matrix and its stadistics
function printConfusionMatrix(matrix::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Matrix{Int64}})
    println("-----------------------\r")
    println("-------|   -   |   +   \r")
    println("-----------------------\r")
    println("   +   |   ", matrix[8][1,1], "   |   ", matrix[8][1,2], "    \r")
    println("-----------------------\r")
    println("   -   |   ", matrix[8][2,1], "   |   ", matrix[8][2,2], "    \r")
    println("-----------------------\r")
    println("Accuracy: ", matrix[1])
    println("Error rate: ", matrix[2])
    println("Sensitivity: ", matrix[3])
    println("Specificity: ", matrix[4])
    println("Positive Prediction Value: ", matrix[5])
    println("Negative Prediction Value: ", matrix[6])
    println("Fscore: ", matrix[7])
end

# Prints a generic confusion Matrix and its stadistics
function printConfusionMatrix(matrix::Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Matrix{Float64}})
    println("-----------------------\r")
    println("-------|   -   |   +   \r")
    println("-----------------------\r")
    println("   +   |   ", matrix[8][1,1], "   |   ", matrix[8][1,2], "    \r")
    println("-----------------------\r")
    println("   -   |   ", matrix[8][2,1], "   |   ", matrix[8][2,2], "    \r")
    println("-----------------------\r")
    println("Accuracy: ", matrix[1])
    println("Error rate: ", matrix[2])
    println("Sensitivity: ", matrix[3])
    println("Specificity: ", matrix[4])
    println("Positive Prediction Value: ", matrix[5])
    println("Negative Prediction Value: ", matrix[6])
    println("Fscore: ", matrix[7])
end

# Prints the confusion Matrix and its stadistics
function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    res = confusionMatrix(outputs, targets)
    
    printConfusionMatrix(res)
end

# Prints the confusion matrix
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    res = confusionMatrix(outputs, targets, threshold= threshold)
    
    printConfusionMatrix(res)
end

# returns a vector of length N, where each element indicates in which subset that pattern should be included
# N= Number of patterns, k= number of subsets into which the dataset is to be split
function crossvalidation(N::Int64, k::Int64)
    #folds = collect(1:k) # Vector with the k folds
    # to make the function repeatable
    Random.seed!(2)
    #indices = repeat(folds, outer=Int(ceil(N/k)));
    indices = repeat(1:k, Int64(ceil(N/k)))
    
    # Select first N indexes
    indices =  indices[1:N]
    
    # Shuffle indexes
    indices = shuffle(indices)
    
    return indices;
end

# Returns the cross validation indexes
# targets: desired outputs, k: number of subsets in which the dataset will be split
# returns a vector of length N (equal to the number of rows of targets)
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #Create a vector of indices, with as many values as rows in the target matrix.
    indices = zeros(size(targets,1))
    
    #Write a loop that iterates over the classes (columns in the target matrix), and does the following:
    numClasses = size(targets, 2)
    indices = Array{Int64,1}(undef,size(targets,1));
    
    for numClass in 1:numClasses
        indices[targets[:, numClass]]=crossvalidation(sum(targets[:, numClass]), k)
    end
    
    # Returns a vector of length N (equal to the number of rows of targets)
    return indices
end

# Returns the cross validation indexes
# targets: desired outputs, k: number of subsets in which the dataset will be split
# returns a vector of length N (equal to the number of rows of targets)
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets)
    numClasses = length(classes)
    indices = Array{Int64,1}(undef,length(targets))
    
    for class in classes
        indicesThisClass = (targets .== class)
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass),k)
    end
    
    # Encode to binary
    #encoded_targets = oneHotEncoding(targets, unique(targets))
    
    #return crossvalidation(encoded_targets, k)
    
    return indices
end

# Creates the model, trains it and return the cross validation stadistics
function modelCrossValidation(modelType::Symbol,
    modelHyperParameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

    # Load inputs and targets
    numFolds = maximum(crossValidationIndices);

    model = undef

    #Create a vector with k elements, which will contain the test results of the cross-validation process 
    # with the selected metric. If more than one metric is to be used, create one vector per metric
    crossvalidationResults_accuracy = Vector{Float64}(undef, numFolds);
    crossvalidationResults_errorrate = Vector{Float64}(undef, numFolds);
    crossvalidationResults_sensitivity = Vector{Float64}(undef, numFolds);
    crossvalidationResults_specificity = Vector{Float64}(undef, numFolds);
    crossvalidationResults_fscore = Vector{Float64}(undef, numFolds);

    #ANN is defined inside the loop of folds
    if (modelType == :SVM)
        # Additional optional parameters
        coef0 = get(modelHyperParameters, "coef0", 0.0)
        shrinking = get(modelHyperParameters, "shrinking", true)
        probability = get(modelHyperParameters, "probability", false)
        tol = get(modelHyperParameters, "tol", 0.001)

        model = SVC(kernel=modelHyperParameters["kernel"], degree=modelHyperParameters["kernelDegree"], 
        gamma = modelHyperParameters["kernelGamma"], C=modelHyperParameters["C"], tol=tol);
    elseif (modelType == :DecisionTree)
        # Additional optional parameters
        criterion = get(modelHyperParameters, "criterion", "gini")
        splitter = get(modelHyperParameters, "splitter", "best")
        min_samples_split = get(modelHyperParameters, "min_samples_split", 2)

        # Decision trees
        # Maximum tree depth
        model = DecisionTreeClassifier(max_depth=modelHyperParameters["max_depth"], random_state=modelHyperParameters["random_state"],
            criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
    elseif (modelType == :kNN)
        # Additional optional parameters
        weights = get(modelHyperParameters, "weights", "uniform")
        metric = get(modelHyperParameters, "metric", "nan_euclidean")

        # kNN
        # k (number of neighbours to be considered)
        model = KNeighborsClassifier(modelHyperParameters["n_neighbors"], weights=weights, metric=metric)
    elseif (modelType == :MLP)    
        validationRatio = get(modelHyperParameters, "validationRatio", convert(Float64, 0.0))
        
        if (validationRatio > 0)
            model = MLPClassifier(hidden_layer_sizes=modelHyperParameters["topology"], max_iter=modelHyperParameters["maxEpochs"],
                learning_rate_init=modelHyperParameters["learningRate"],early_stopping=true, validation_fraction=validationRatio)
        else
            model = MLPClassifier(hidden_layer_sizes=modelHyperParameters["topology"], max_iter=modelHyperParameters["maxEpochs"],
                learning_rate_init=modelHyperParameters["learningRate"],early_stopping=true)
        end
    elseif (modelType == :GB)    
        model = GaussianNB()
    elseif (modelType == :LR)    
        max_iter = get(modelHyperParameters,"max_iter",100)
        multi_class = get(modelHyperParameters,"multi_class","multinomial")

        model = LogisticRegression(max_iter=max_iter, multi_class=multi_class)
    elseif (modelType == :NC)    
        model = NearestCentroid()
    elseif (modelType == :RN)    
        model = RadiusNeighborsClassifier()
    elseif (modelType == :RR)    
        model = RidgeClassifier()
    end

    # Make a loop with k iterations (k folds) where, within each iteration, 4 matrices are created 
    # from the desired input and output matrices by means of the index vector resulting from the previous function. 
    # Namely, the desired inputs and outputs for training and test
    for numFold in 1:numFolds
        trainingInputs = inputs[kFoldIndices.!=numFold,:];
        trainingTargets = targets[kFoldIndices.!=numFold,:];
        testInputs = inputs[kFoldIndices.==numFold,:];
        testTargets = targets[kFoldIndices.==numFold,:];
        
        # If the model is an ANN, the desired outputs shall be encoded by means of the code developed in previous assignments. 
        # As this model is non-deterministic, it will be necessary to make a new loop to train several ANNs, 
        # splitting the training data into training and validation (if validation set is used) and calling the function defined 
        if (modelType == :ANN)
            # In the case of ANN training, the desired outputs shall be encoded as done in previous assignments.
            trainingTargets = oneHotEncoding(vec(trainingTargets))
            testTargets = oneHotEncoding(vec(testTargets))
            
            repetitionsTraining=modelHyperParameters["repetitionsTraining"]
            
            # AAN are not deterministic, so we must repeat each fold several times and
            # save the metrics for each of those iterations (for ex., accuracy and f1)
            testANNAccuraciesEachRepetition = Array{Float64,1}(undef, repetitionsTraining); 
            testANNerrorrateEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNsensitivityEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNspecificityEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNF1EachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            
            for numTraining in 1:repetitionsTraining
                validationRatio = modelHyperParameters["validationRatio"]
                
                if validationRatio>0
                    # Train ANN using training, validation and test sets.
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), 
                        validationRatio*size(trainingInputs,1)/size(inputs,1));
                    
                    model, = trainClassANN(modelHyperParameters["topology"],(trainingInputs, trainingTargets),
                        validationDataset=(trainingInputs[validationIndices[2],1:4],trainingTargets[validationIndices[2],:]), 
                        testDataset=(testInputs, testTargets), transferFunctions=modelHyperParameters["transferFunctions"],
                        maxEpochs=modelHyperParameters["maxEpochs"], minLoss=modelHyperParameters["minLoss"], 
                        learningRate=modelHyperParameters["learningRate"], maxEpochsVal=modelHyperParameters["maxEpochsVal"])
                else
                    # Train ANN using training and test sets.
                    model, = trainClassANN(modelHyperParameters["topology"],(trainingInputs, trainingTargets),
                        testDataset=(testInputs, testTargets), transferFunctions=modelHyperParameters["transferFunctions"],
                        maxEpochs=modelHyperParameters["maxEpochs"], minLoss=modelHyperParameters["minLoss"], 
                        learningRate=modelHyperParameters["learningRate"], maxEpochsVal=modelHyperParameters["maxEpochsVal"])
                end
                
                # Get prediction and transform outputs
                outputs = model(testInputs')
                vmax = maximum(outputs', dims=2);
                outputs = (outputs' .== vmax);

                metrics = confusionMatrix(outputs, testTargets, weighted=false)

                # Uncomment to test model accuracy
                #println("Accuracy: ", metrics[1], " Error rate: ", metrics[2]," Sensitivity: ", metrics[3]," Specificity: ", metrics[4]," Fscore: ", metrics[7])
                #realAccuracy(outputs, testTargets)
                
                #return the average of the test results (with the selected metric or metrics) in order to have the test value 
                #corresponding to this k.
                testANNAccuraciesEachRepetition[numTraining] = metrics[1];# Accuracy
                testANNerrorrateEachRepetition[numTraining] = metrics[2];# errorrate
                testANNsensitivityEachRepetition[numTraining] = metrics[3];# sensitivity
                testANNspecificityEachRepetition[numTraining] = metrics[4];# specificity
                testANNF1EachRepetition[numTraining] = metrics[7];# Fscore
            end
            
            #Once the model has been trained (several times) on each fold, take the result and fill in the vector(s) 
            #created earlier (one for each metric).
            crossvalidationResults_accuracy[numFold] = mean(testANNAccuraciesEachRepetition);
            crossvalidationResults_errorrate[numFold] =  mean(testANNerrorrateEachRepetition);
            crossvalidationResults_sensitivity[numFold] = mean(testANNsensitivityEachRepetition);
            crossvalidationResults_specificity[numFold] = mean(testANNspecificityEachRepetition);
            crossvalidationResults_fscore[numFold] = mean(testANNF1EachRepetition);
        else
            #vec to avoid DataConversionWarning ravel
            fit!(model, trainingInputs, vec(trainingTargets));

            testOutputs = predict(model, testInputs);
            metrics = confusionMatrix(testOutputs, vec(testTargets), weighted=false);

            # Uncomment to test model accuracy
            #println("Accuracy: ", metrics[1], " Error rate: ", metrics[2]," Sensitivity: ", metrics[3]," Specificity: ", metrics[4]," Fscore: ", metrics[7])
            #realAccuracy(testOutputs, vec(testTargets))
            
            #Once the model has been trained (several times) on each fold, take the result and fill in the vector(s) 
            #created earlier (one for each metric).
            crossvalidationResults_accuracy[numFold] = metrics[1];# Accuracy
            crossvalidationResults_errorrate[numFold] = metrics[2];# errorrate
            crossvalidationResults_sensitivity[numFold] = metrics[3];# sensitivity
            crossvalidationResults_specificity[numFold] = metrics[4];# specificity
            crossvalidationResults_fscore[numFold] = metrics[7];# Fscore
        end #if type of model
    end #for each fold

    #Finally, provide the result of averaging the values of these vectors for each metric together 
    #with their standard deviations.
    accuracy = mean(crossvalidationResults_accuracy)
    errorrate = mean(crossvalidationResults_errorrate)
    sensitivity = mean(crossvalidationResults_sensitivity)
    specificity = mean(crossvalidationResults_specificity)
    fscore = mean(crossvalidationResults_fscore)

    #As a result of this call, at least the test value in the selected metric(s) should be returned. 
    #If the model is not deterministic (as is the case for the ANNs), it will be the average of the results of several trainings.
    return (model, [accuracy, errorrate, sensitivity, specificity, fscore]);
end

# Train an ensemble of models
function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperParameters::AbstractArray{Dict, 1},     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
        kFoldIndices::Array{Int64,1})

    @assert length(estimators) == length(modelsHyperParameters)

    # Load inputs and targets
    numFolds = maximum(kFoldIndices);

    #Create a vector with k elements, which will contain the test results of the accuracy process 
    crossvalidationResults_accuracy = Vector{Float64}(undef, numFolds);
        
    for numFold in 1:length(unique(kFoldIndices))
        # Prepare the data in train test split with CV indices
        println("Procesing Fold: ", numFold)
        
        trainingInputs = trainingDataset[1][kFoldIndices.!=numFold,:];
        trainingTargets = trainingDataset[2][kFoldIndices.!=numFold,:];
        testInputs = trainingDataset[1][kFoldIndices.==numFold,:];
        testTargets = trainingDataset[2][kFoldIndices.==numFold,:];
        
        fold_estimators = []
        for (index,estimator) in enumerate(estimators)
            #Create the models and store in array
            if (estimator == :SVM)
                # Additional optional parameters
                coef0 =  haskey(modelsHyperParameters[index], "coef0") ? modelsHyperParameters[index]["coef0"] : 0.0
                shrinking = haskey(modelsHyperParameters[index], "shrinking") ? modelsHyperParameters[index]["shrinking"] : true
                probability = haskey(modelsHyperParameters[index], "probability") ? modelsHyperParameters[index]["probability"] : false
                tol = haskey(modelsHyperParameters[index], "tol") ? modelsHyperParameters[index]["tol"] : 0.001

                fold_model = SVC(kernel=modelsHyperParameters[index]["kernel"], degree=modelsHyperParameters[index]["kernelDegree"], 
                gamma = modelsHyperParameters[index]["kernelGamma"], C=modelsHyperParameters[index]["C"], tol=tol);
            elseif (estimator == :DecisionTree)
                # Additional optional parameters
                criterion = haskey(modelsHyperParameters[index], "criterion") ? modelsHyperParameters[index]["criterion"] : "gini"
                splitter = haskey(modelsHyperParameters[index], "splitter") ? modelsHyperParameters[index]["splitter"] : "best"
                min_samples_split = haskey(modelsHyperParameters[index], "min_samples_split") ? modelsHyperParameters[index]["min_samples_split"] : 2

                # Decision trees
                # Maximum tree depth
                fold_model = DecisionTreeClassifier(max_depth=modelsHyperParameters[index]["max_depth"], random_state=modelsHyperParameters[index]["random_state"],
                criterion=criterion, splitter=splitter, min_samples_split=min_samples_split);
            elseif (estimator == :kNN)
                # Additional optional parameters
                weights = haskey(modelsHyperParameters[index], "weights") ? modelsHyperParameters[index]["weights"] : "uniform"
                metric = haskey(modelsHyperParameters[index], "metric") ? modelsHyperParameters[index]["metric"] : "nan_euclidean"

                # kNN
                # k (number of neighbours to be considered)
                fold_model = KNeighborsClassifier(modelsHyperParameters[index]["n_neighbors"], weights=weights, metric=metric);
            elseif (estimator == :ANN)
                validationRatio = haskey(modelsHyperParameters[index], "validationRatio") ? modelsHyperParameters[index]["validationRatio"] : 0
                
                # (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(trainingDataset[1],1));
                
                if validationRatio > 0
                    fold_model = MLPClassifier(hidden_layer_sizes=modelsHyperParameters[index]["topology"], max_iter=modelsHyperParameters[index]["maxEpochs"],
                        learning_rate_init=modelsHyperParameters[index]["learningRate"],early_stopping=true, validation_fraction=validationRatio)
                else
                    fold_model = MLPClassifier(hidden_layer_sizes=modelsHyperParameters[index]["topology"], max_iter=modelsHyperParameters[index]["maxEpochs"],
                        learning_rate_init=modelsHyperParameters[index]["learningRate"])
                end
            end
            
            # vec to avoid DataConversionWarning ravel
            fit!(fold_model, trainingInputs, vec(trainingTargets));

            # Add model to vector of estimators
            push!(fold_estimators,(string("Model_", index), fold_model))
        end # for each estimator

        ensemble_model = VotingClassifier(estimators = [(x[1],x[2]) for x in fold_estimators], n_jobs=-1)

        # Fit the ensemble model and predict using the test set
        fit!(ensemble_model, trainingInputs, vec(trainingTargets))

        # Ver si usar la matriz de confusion o el valor que dan en la unidad score(model,test_input, test_output)
        acc = score(ensemble_model, testInputs, testTargets)

        crossvalidationResults_accuracy[numFold] = acc;
    end # for each fold
    
    # Finally, provide the result of averaging the values of these vectors for each metric together with their standard deviations.
    accuracy = mean(crossvalidationResults_accuracy)

    return (accuracy)
end

# Check manually the real accuracy value
function realAccuracy(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1})
    ok=0
    tot=0
    for x in 1:size(outputs,1)
        tot = tot + 1
        if (outputs[x]==targets[x])
            ok = ok + 1
        end
    end

    println("Tot: ", tot, " Ok: ", ok, " Acc: ", ok/tot)
end

# Check manually the real accuracy value
function realAccuracy(outputs::AbstractArray{<:Bool,2}, targets::AbstractArray{<:Bool,2})
    ok=0
    tot=0
    for x in 1:size(outputs,1)
        tot = tot + 1
        if (outputs[x,:]==targets[x,:])
            ok = ok + 1
        end
    end

    println("Tot: ", tot, " Ok: ", ok, " Acc: ", ok/tot)
end