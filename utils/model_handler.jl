# Test for the best ANN Model
function test_ANN_Model(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},    
    kFoldIndices::Array{Int64,1})
    parameters = Dict();

    # For ANNs, test at least 8 different architectures, between one and 2 hidden layers.
    parameters["maxEpochs"] = 1000
    parameters["minLoss"] = 0.0
    parameters["learningRate"] = 0.01
    parameters["repetitionsTraining"] = 5
    parameters["maxEpochsVal"] = 20
    parameters["validationRatio"] = 0

    parameters["topology"] = [8,3,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, (convert(Float64, 0), Dict()))

    parameters["topology"] = [8,4,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [14,7,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [16,8,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [28,14,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [32,16,8,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [28,14,7,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [16,8,4,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["topology"] = [20,10,5,1]
    parameters["transferFunctions"] = fill(logsigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    # Assign the best topology of previous test to check some others hyperparameters
    parameters["topology"] = res[2]["topology"];

    # Learning rate   
    parameters["learningRate"] = 0.1
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["learningRate"] = 0.001
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["learningRate"] = 0.01

    # Transfer functions
    parameters["transferFunctions"] = fill(sigmoid, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    parameters["transferFunctions"] = fill(tanh_fast, length(parameters["topology"]))
    res = evaluateModel(:ANN, parameters, inputs, targets, kFoldIndices, res)

    println("Best parameters: ", res[2])
    println("Best accuracy: ", res[1])
end

# Test for the best SVM Model
function test_SVM_Model(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},    
    kFoldIndices::Array{Int64,1})
    parameters = Dict();

    # For SVM, test with different kernels and values of C. At least 8 SVM hyperparameter configurations.
    parameters["kernel"] = "rbf";
    parameters["kernelDegree"] = 3;
    parameters["kernelGamma"] = 2;
    parameters["C"] = 1;

    # Additional optional parameters
    parameters["coef0"] =  0.0
    parameters["shrinking"] = true
    parameters["probability"] = false
    parameters["tol"] = 0.001
    
    println("Test results for SVM model: ")

    # Test combination of Kernel and C values
    parameters["kernel"] = "rbf";
    parameters["C"] = 1;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, (convert(Float64, 0), Dict()))
    
    parameters["C"] = 2;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 10;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)

    parameters["kernel"] = "linear";
    parameters["C"] = 1;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 2;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 10;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["kernel"] = "poly";
    parameters["C"] = 1;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 2;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 10;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["kernel"] = "sigmoid";
    parameters["C"] = 1;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 2;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["C"] = 10;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    # Assign the best kernel and C of previous test to check the rest of the hyperparameters
    parameters["kernel"] = res[2]["kernel"];
    parameters["C"] = res[2]["C"];

    # Degree test   
    parameters["kernelDegree"] = 5;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
 
    parameters["kernelDegree"] = 1;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["kernelDegree"] = 3;

    # Gamma test
    parameters["kernelGamma"] = 3;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    # Tolerance for stopping criterion
    parameters["tol"] = 0.01;
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
    println("Best parameters: ", res[2])
    println("Best accuracy: ", res[1])
end

# Test for the best decision tree Model
function test_DT_Model(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},    
    kFoldIndices::Array{Int64,1})
    parameters = Dict();

    # For decision trees, test at least 6 different depth values.
    parameters["max_depth"]=4
    parameters["random_state"]=1

    # Additional optional parameters
    parameters["criterion"] = "gini"
    parameters["splitter"] = "best"
    parameters["min_samples_split"] = 2
    
    println("Test results for Decision tree model: ")

    # Test max depth values
    parameters["max_depth"]=4
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, (convert(Float64, 0), Dict()))
    
    parameters["max_depth"]=3
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["max_depth"]=2
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)

    parameters["max_depth"]=1
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["max_depth"]=5
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["max_depth"]=6
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    # Assign the best kernel and C of previous test to check the rest of the hyperparameters
    parameters["max_depth"] = res[2]["max_depth"];

    # Splitter  
    parameters["splitter"] = "random";
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)

    parameters["splitter"] = "best"
 
    # min_samples_split
    parameters["min_samples_split"] = 4;
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["min_samples_split"] = 3;
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)

    parameters["min_samples_split"] = 1;
    res = evaluateModel(:DecisionTree, parameters, inputs, targets, kFoldIndices, res)
    
    println("Best parameters: ", res[2])
    println("Best accuracy: ", res[1])
end

# Test for the best KNN Model
function test_KNN_Model(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},    
    kFoldIndices::Array{Int64,1})
    parameters = Dict();

    # For kNN, test at least 6 different k values.
    parameters["n_neighbors"]=3

    # Additional optional parameters
    parameters["weights"] = "uniform"
    parameters["metric"] = "nan_euclidean"
    
    println("Test results for KNN model: ")

    # Test max depth values
    parameters["n_neighbors"]=3
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, (convert(Float64, 0), Dict()))
    
    parameters["n_neighbors"]=2
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["n_neighbors"]=1
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)

    parameters["n_neighbors"]=4
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["n_neighbors"]=5
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)
    
    parameters["n_neighbors"]=6
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)
    
    # Assign the best k of previous test to check the rest of the hyperparameters
    parameters["n_neighbors"] = res[2]["n_neighbors"];

    # weights  
    parameters["weights"] = "distance";
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)

    parameters["weights"] = "uniform"
 
    # metric
    parameters["metric"] = "minkowski";
    res = evaluateModel(:kNN, parameters, inputs, targets, kFoldIndices, res)
    
    println("Best parameters: ", res[2])
    println("Best accuracy: ", res[1])
end

function evaluateModel(modelType::Symbol,
    modelHyperParameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1}, 
    previousModel::Tuple{Float64, Dict})

    model_accuracy = modelCrossValidation(modelType, modelHyperParameters, inputs, targets, crossValidationIndices)
    # return (model, [accuracy, errorrate, sensitivity, specificity, fscore]);
    println("Parameters: ", modelHyperParameters)
    println("Accuracy: ", model_accuracy[2][1])

    if (model_accuracy[2][1] > previousModel[1])
        best_parameters = modelHyperParameters
        best_accuracy = model_accuracy[2][1]
    else
        best_parameters = previousModel[2]
        best_accuracy = previousModel[1]
    end

    return (best_accuracy, best_parameters)
end