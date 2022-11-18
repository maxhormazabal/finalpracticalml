using Flux;
using Flux.Losses;
using DelimitedFiles;
using Statistics;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
   
    return convert(NTuple{2, AbstractArray{<:Real,2}}, (mins, maxs))
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    
    #Get maximum and minimun as vector for use in comparison
    max = vec(normalizationParameters[2])
    min = vec(normalizationParameters[1])
    
    #Normalization formula v'=(v-min)/(max-min). If max equals to min, no normalization is required so constants are used.
    dataset = (dataset.-(max == min ? 0 : normalizationParameters[1]))./(max == min ? 1 : normalizationParameters[2]-normalizationParameters[1])
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    
    ds = copy(dataset)
    normalizeMinMax!(ds, normalizationParameters)
    return ds
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    ds = copy(dataset)
    
    normalizeMinMax(ds, calculateMinMaxNormalizationParameters(ds))
    return ds
end

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

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return oneHotEncoding(feature, unique(feature))
end

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
        #If no layer, use an identity
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));   
    end
    
    if numOutputs > 1
        ann = Chain(ann...,  softmax)
	else
		ann = Chain(ann...,  identity)
    end
    
    return ann;
end

#N=number of patterns, P=Percentage of patters separated
# by using randperm along side with usign random, we guarantee that 
function holdOut(N::Int, P::Real)
#to assure that the function have repeatable results for splitting, Random.seed! was used inside the function and assign any constant value
    Random.seed!(2)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N)
    n_train = Int(round((1 - P)*N))
    return (indices[1:n_train],indices[n_train+1:end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #Check that Pval and Ptest contain values between 0 and 1,
    # as well as the sum of both values should be lower or equal to 1.
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest<=1.));
    
    # First, separate dataset into training+validation and test using previous holdOut method
    dataset_tr_val_tst = holdOut(N, Ptest);

    # You must also adjust the ratio applied in the second call of the function.
    # For example, it is not the same Ptest=0.2 over the whole dataset than over just the training+validation set. You have
    # to "adjust" it.
    # Remove test information from the variable and recalculate
    N_tr_val=N-length(dataset_tr_val_tst[2])
    Pval_tr_val=N*Pval/N_tr_val
    
    # Then, separate training+validation into training and validation using again previous holdOut method.
    dataset_tr_val = holdOut(N_tr_val, Pval_tr_val);
    
    dataset_tr = dataset_tr_val_tst[1][1:length(dataset_tr_val[1])]
    dataset_val = dataset_tr_val_tst[1][length(dataset_tr_val[1])+1:length(dataset_tr_val_tst[1])]
      
    # Keep in mind that the indexes from the return tuple must be used to obtain the elements from the training+validation set.
    # Otherwise, you will have repeated indexes in the sets.
    return (dataset_tr,dataset_val,dataset_tr_val_tst[2])
end

function calculateLossValues()
    trainingLoss = loss(trainingInputs', trainingTargets');
    validationLoss = loss(validationInputs', validationTargets');
    testLoss = loss(testInputs', testTargets');
    #(trainingLoss, validationLoss, _) = calculateLossValues();
    push!(trainingLosses, trainingLoss);
    push!(validationLosses, validationLoss);
    push!(testLosses, testLoss);
end

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
    
    if (FN+TP) > 0
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
        
    if (TP+FP) > 0
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

# Calculate the confusion matrix and its stadistics
function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs.>= threshold;

    return confusionMatrix(binary_outputs, targets)
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

#Receives two lists of boolean and returns the average of equal values between them
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    tot = length(outputs[outputs .== targets])
    return tot / length(outputs)
end

#Receives two lists of boolean and returns the average of equal values between them
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    if (size(outputs)[:2]>1)
        classComparison = targets .!= outputs 
        incorrectClassifications = any(classComparison, dims=2)
        return 1 - mean(incorrectClassifications)
    else
        accuracy(outputs[1], targets[1])
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    
    #A vector with the boolean classification
    result = Vector{Bool}(undef, length(outputs));
        
    #Compare each value to the threshold
    result.= outputs.>= threshold;
        
    tot = length(result[result .== targets])
    return tot / length(result)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    
    if (size(outputs)[:2]>1)
        values = classifyOutputs(outputs);
        accuracy(values);
    else
        accuracy(outputs[1]);
    end    
end

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
    # The first thing this function should do is to check that the number of columns of both matrices is equal 
    @assert(size(outputs, 2)==size(targets, 2));
    # and is different from 2. 
    @assert(size(outputs, 2)!=2);
    @assert(size(targets, 2)!=2);
    
    # In case they have only one column, these columns are taken as vectors and the confusion Matrix function 
    # developed in the previous assignment is called.
    if (size(outputs, 2)==1)
        return confusionMatrix(vec(outputs), vec(targets))
    else
        #Gets the number of classes
        #numClasses = size(unique(targets, dims=1),1)
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
                # println(outputs_class)
                # println(targets_class)
                
                # make a call to the confusionMatrix function of the previous assignment 
                # passing as vectors the columns corresponding to the class of that iteration of the outputs and targets matrices. 
                res = confusionMatrix(vec(outputs_class), vec(targets_class))
                # printConfusionMatrix(res)
                
                #Assign the result to the corresponding element of the sensitivity, specificity, PPV, NPV and F1 vectors.
                stadistics[numClass,1] = res[3] # Sensitivity
                stadistics[numClass,2] = res[4] # Specificity
                stadistics[numClass,3] = res[5] # PPV
                stadistics[numClass,4] = res[6] # NPV
                stadistics[numClass,5] = res[7] # F-score
            end
        end
        
        # println("Stadistics: ", stadistics)
        
        #Reserve memory for the confusion matrix.
        confusion_matrix = zeros(numClasses, numClasses)
        
        #Perform a double loop in which booth loops iterate over the classes, to fill all the confusion matrix elements.
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
        
        # println("Partial confusion matrix: ",confusion_matrix)
        
        #Aggregate the values of sensitivity, specificity, PPV, NPV, and F-score for eachclass into a single value.
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

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    values = classifyOutputs(outputs);
    
    confusionMatrix(values, targets, weighted = weighted) 
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # it is necessary that all the output classes (vector outputs) are included in the desired output classes (vector targets) 
    @assert(all([in(output, unique(targets)) for output in outputs]))
    
    # Use the same list of classes for encoding the same way
    encoded_targets = oneHotEncoding(targets, unique(targets))
    encoded_outputs = oneHotEncoding(outputs, unique(targets))
      
    return confusionMatrix(encoded_outputs, encoded_targets, weighted=weighted)
end

# Unit 5


# returns a vector of length N, where each element indicates in which subset that pattern should be included
# N= Number of patterns, k= number of subsets into which the dataset is to be split
function crossvalidation(N::Int64, k::Int64)
    #folds = collect(1:k) # Vector with the k folds
    
    #indices = repeat(folds, outer=Int(ceil(N/k)));
    indices = repeat(1:k, Int64(ceil(N/k)))
    
    # Select first N indexes
    indices =  indices[1:N]
    
    # Shuffle indexes
    indices = shuffle(indices)
    
    return indices;
end

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
        # Take the number of elements belonging to that class. 
        # This can be done by making a call to the sum function applied to the corresponding column.
        #num_elements = sum(targets[:,[numClass]])
        #@assert(num_elements>=k)
        # Call to the crossvalidation passing as parameters this number of elements and the value of k.
        #indexs = crossvalidation(num_elements, k)

        #Update the index vector positions indicated by the corresponding column of the targets matrix 
        #with the values of the vector resulting from the call to the crossvalidation function.
        #y=1
        #targets_class=targets[:,[numClass]][:]
        #for x in 1:size(targets_class,1)
        #    if (targets_class[x]==true)
        #        indices[x]=indexs[y]
        #        y=y+1
        #    end
        #end
    end
    
    # Returns a vector of length N (equal to the number of rows of targets)
    return indices
end

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

# Creates the model, trains it and return the stadistics
function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
    
    
    # The developed normalisation functions should also be used on the data to be used by these models
    normalizedinputs = normalizeMinMax(inputs)
    
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
        coef0 =  haskey(parameters, "coef0") ? parameters["coef0"] : 0.0
        shrinking = haskey(parameters, "shrinking") ? parameters["shrinking"] : true
        probability = haskey(parameters, "probability") ? parameters["probability"] : false
        tol = haskey(parameters, "tol") ? parameters["tol"] : 0.001

        model = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], 
            gamma = parameters["kernelGamma"], C=parameters["C"], tol=tol);
    elseif (modelType == :DecisionTree)
        # Additional optional parameters
        criterion = haskey(parameters, "criterion") ? parameters["criterion"] : "gini"
        splitter = haskey(parameters, "splitter") ? parameters["splitter"] : "best"
        min_samples_split = haskey(parameters, "min_samples_split") ? parameters["min_samples_split"] : 2

        # Decision trees
        # Maximum tree depth
        model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=parameters["random_state"],
            criterion=criterion, splitter=splitter, min_samples_split=min_samples_split);
    elseif (modelType == :kNN)
        # Additional optional parameters
        weights = haskey(parameters, "weights") ? parameters["weights"] : "uniform"
        metric = haskey(parameters, "metric") ? parameters["metric"] : "nan_euclidean"

        # kNN
        # k (number of neighbours to be considered)
        model = KNeighborsClassifier(parameters["n_neighbors"], weights=weights, metric=metric);
    end
    
    # Make a loop with k iterations (k folds) where, within each iteration, 4 matrices are created 
    # from the desired input and output matrices by means of the index vector resulting from the previous function. 
    # Namely, the desired inputs and outputs for training and test
    for numFold in 1:numFolds
        #println("Procesing Fold: ", numFold)
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
            
            repetitionsTraining=parameters["repetitionsTraining"]
            
            # AAN are not deterministic, so we must repeat each fold several times and
            # save the metrics for each of those iterations (for ex., accuracy and f1)
            testANNAccuraciesEachRepetition = Array{Float64,1}(undef, repetitionsTraining); 
            testANNerrorrateEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNsensitivityEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNspecificityEachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            testANNF1EachRepetition = Array{Float64,1}(undef, repetitionsTraining);
            
            for numTraining in 1:repetitionsTraining
                validationRatio = parameters["validationRatio"]
                
                if validationRatio>0
                    # Train ANN using training, validation and test sets.
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), 
                        validationRatio*size(trainingInputs,1)/size(inputs,1));
                    
                    model, = trainClassANN(parameters["topology"],(trainingInputs, trainingTargets),
                        validationDataset=(trainingInputs[validationIndices[2],1:4],trainingTargets[validationIndices[2],:]), 
                        testDataset=(testInputs, testTargets), transferFunctions=parameters["transferFunctions"],
                        maxEpochs=parameters["maxEpochs"], minLoss=parameters["minLoss"], 
                        learningRate=parameters["learningRate"], maxEpochsVal=parameters["maxEpochsVal"])
                else
                    # Train ANN using training and test sets.
                    model, = trainClassANN(parameters["topology"],(trainingInputs, trainingTargets),
                        testDataset=(testInputs, testTargets), transferFunctions=parameters["transferFunctions"],
                        maxEpochs=parameters["maxEpochs"], minLoss=parameters["minLoss"], 
                        learningRate=parameters["learningRate"], maxEpochsVal=parameters["maxEpochsVal"])
                end
                
                # Get prediction and transform outputs
                outputs = model(testInputs')
                vmax = maximum(outputs', dims=2);
                outputs = (outputs' .== vmax);

                metrics = confusionMatrix(outputs, testTargets, weighted=false)
                # printConfusionMatrix(metrics);

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