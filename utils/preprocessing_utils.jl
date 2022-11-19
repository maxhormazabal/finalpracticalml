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
