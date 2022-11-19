# Normalize time in minutes to miliseconds
function normalizeMinutesMiliseconds(columnToNormalize::Any)
    for i in (1:length(columnToNormalize))
        # If the value is less than 100, its in minutes
        if columnToNormalize[i] < 100
            # Multiply the value to convert from minutes to miliseconds
            columnToNormalize[i] = columnToNormalize[i] * 60000
        end
    end

    return columnToNormalize
end


# Feature Selection
function FeatureSelection(inputs,outputs,k)

    selector = UnivariateFeatureSelector(method=pearson_correlation, k=k)
    UnivariateFeatureSelector(FeatureSelectors.pearson_correlation, 5, nothing)

    Features = select_features(selector, DataFrame(inputs, :auto),vec(outputs))

    return(Features)

end

