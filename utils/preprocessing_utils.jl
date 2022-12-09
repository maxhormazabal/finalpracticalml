# Convert numeric values into string not considering missing values
function toString(x)
    if ismissing(x)==true
        return missing
    elseif ismissing(x)==false
        return string(x)
    end
end

# Calculate ranges of outliers and extreme outliers 
function quartilOutliers(data)
    Q1 = quantile(data,0.25)
    Q3 = quantile(data,0.75)
    RIC = Q3 -Q1
    sup_outlier = Q3+1.5*RIC
    inf_outlier = Q1-1.5*RIC
    ext_sup_outlier = Q3+3*RIC
    ext_inf_outlier = Q1-3*RIC

    return (sup_outlier,inf_outlier,ext_sup_outlier,ext_inf_outlier)
end

# Convert to a table
function pTable(s)
    d = Dict()
    for c in s
        if c ∉ keys(d)
            d[c] = 1
        else
            d[c] += 1
        end
    end
    array = convert(Matrix{Any},zeros(length(d),2))
    keys_dict = keys(d)

    for (index,name) in enumerate(keys_dict)
        array[index,1] = name
        array[index,2] = d[name]
    end
    return (d,array)
end

# For linear regression in loudness
function predLoudness(value)
    x = value
    x2 = value^2
    x3 = value^3
    y = -20.8171 + 41.27*x - 48.2937*x2 + 23.8144*x3
    return y
end

# Normalize time in minutes to miliseconds
function durationToSeconds(columnToNormalize::Any)
    for i in (1:length(columnToNormalize))
        # If the value is less than 100, its in minutes
        if columnToNormalize[i] < 100
            # Multiply the value to convert from minutes to miliseconds
            columnToNormalize[i] = columnToNormalize[i] * 60000
        end
    end
    
    for i in (1:length(columnToNormalize))
        # If the value is less than 100, its in minutes
        if columnToNormalize[i] > 100
            # Multiply the value to convert from minutes to miliseconds
            columnToNormalize[i] = columnToNormalize[i] * 0.001
        end
    end
    return columnToNormalize
end

# Convert categorical column to numeric
function categoricalToNumericDataFrame(column,colname)
    column_dict = Dict()
    for value in levels(column)
        colValues = (column .== value)
        name = string(colname,value)
        column_dict[name] = colValues
    end
    return DataFrame(column_dict)
end

# Run pca transformation on a dataframe
function pcaTranformation(pca_df)
    numrow = size(pca_dataframe,1)
    numcol = size(pca_dataframe,2)
    
    indices = holdOut(numrow,0.2)
    train_input = pca_dataframe[indices[1],1:numcol-1]
    train_output = vec(pca_dataframe[indices[1],"Class"])
    
    test_input = pca_dataframe[indices[2],1:numcol-1]
    test_output = vec(pca_dataframe[indices[2],"Class"])
    
    train_input = normalizeMinMax!(Matrix(train_input))
    test_input = normalizeMinMax!(Matrix(test_input))
    
    pca = PCA(0.95)
    pca.fit(train_input)
    
    pca_train = pca.transform(train_input)
    pca_test = pca.transform(test_input)
    
    println("Train Patterns ", size(train_input), " -> ", size(pca_train))
    println("Test Patterns ", size(test_input), " -> ", size(pca_test))

    return (train_input,train_output,test_input,test_output)
end