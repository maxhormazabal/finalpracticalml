# 1. For ANNs, test at least 8 different architectures, between one and 2 hidden layers.
# 2. For SVM, test with different kernels and values of C. At least 8 SVM hyperparameter configurations.
# 3. For decision trees, test at least 6 different depth values.
# 4. For kNN, test at least 6 different k values.

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
    res = evaluateModel(:SVM, parameters, inputs, targets, kFoldIndices, res)
    
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