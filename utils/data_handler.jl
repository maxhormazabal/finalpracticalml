function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
   
    return convert(NTuple{2, AbstractArray{<:Real,2}}, (mins, maxs))
end
