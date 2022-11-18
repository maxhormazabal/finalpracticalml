using Flux;

# Test for the best SVM Model
function test_SVM_Model(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::Array{Int64,1})
    parameters = Dict();

    #SVM
    parameters["kernel"] = "rbf";
    parameters["kernelDegree"] = 3;
    parameters["kernelGamma"] = 2;
    parameters["C"] = 1;

    # Additional optional parameters
    coef0 =  0.0
    shrinking = true
    probability = false
    tol = 0.001
    
    println("Test results for SVM model: ")
    # Basic model
    model_accuracy = train_Model(:SVM, parameters, trainingDataset, kFoldIndices)
    println(parameters)
    println(model_accuracy)
end

# Train a model and get its accuracy with cross validation
function train_Model(modelType::Symbol, modelsHyperParameters::Dict,     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::Array{Int64,1})

    # Load inputs and targets
    numFolds = maximum(kFoldIndices);
    crossvalidationResults_accuracy = Vector{Float64}(undef, numFolds);

    for numFold in 1:length(unique(kFoldIndices))
        println("Procesing Fold: ", numFold)
        
        trainingInputs = trainingDataset[1][kFoldIndices.!=numFold,:];
        trainingTargets = trainingDataset[2][kFoldIndices.!=numFold,:];
        testInputs = trainingDataset[1][kFoldIndices.==numFold,:];
        testTargets = trainingDataset[2][kFoldIndices.==numFold,:];

        if (modelType == :SVM)
            # Additional optional parameters
            coef0 = get(modelsHyperParameters, "coef0", 0.0)
            shrinking = get(modelsHyperParameters, "shrinking", true)
            probability = get(modelsHyperParameters, "probability", false)
            tol = get(modelsHyperParameters, "tol", 0.001)

            fold_model = SVC(kernel=modelsHyperParameters["kernel"], degree=modelsHyperParameters["kernelDegree"], 
                gamma = modelsHyperParameters["kernelGamma"], C=modelsHyperParameters["C"], tol=tol,
                probability = probability, coef0 = coef0, shrinking= shrinking);
        elseif (modelType == :DecisionTree)
            # Additional optional parameters
            criterion = get(modelsHyperParameters, "criterion", "gini")
            splitter = get(modelsHyperParameters, "splitter", "best")
            min_samples_split = get(modelsHyperParameters, "min_samples_split", 2)

            # Decision trees
            # Maximum tree depth
            fold_model = DecisionTreeClassifier(max_depth=modelsHyperParameters["max_depth"], 
                random_state=modelsHyperParameters["random_state"],
                criterion=criterion, splitter=splitter, min_samples_split=min_samples_split);
        elseif (modelType == :kNN)
            # Additional optional parameters
            weights = get(modelsHyperParameters, "weights", "uniform")
            metric = get(modelsHyperParameters, "metric", "nan_euclidean")

            # kNN
            # k (number of neighbours to be considered)
            fold_model = KNeighborsClassifier(modelsHyperParameters["n_neighbors"], weights=weights, metric=metric);
        elseif (modelType == :ANN)
            validationRatio = get(modelsHyperParameters, "validationRatio", 0)
            
            if validationRatio > 0
                fold_model = MLPClassifier(hidden_layer_sizes=modelsHyperParameters["topology"], max_iter=modelsHyperParameters["maxEpochs"],
                    learning_rate_init=modelsHyperParameters["learningRate"],early_stopping=true, validation_fraction=validationRatio)
            else
                fold_model = MLPClassifier(hidden_layer_sizes=modelsHyperParameters["topology"], max_iter=modelsHyperParameters["maxEpochs"],
                    learning_rate_init=modelsHyperParameters["learningRate"])
            end
        end
        
        # vec to avoid DataConversionWarning ravel
        fit!(fold_model, trainingInputs, vec(trainingTargets));

        acc = score(fold_model, testInputs, testTargets)

        crossvalidationResults_accuracy[numFold] = acc;
    end

    accuracy = mean(crossvalidationResults_accuracy)
    
    return accuracy
end