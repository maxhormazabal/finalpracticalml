# Normalize time in minutes to miliseconds
function normalizeMinutesMiliseconds(columnToNormalize::Any)
    for i in (1:length(columnToNormalize))
        if columnToNormalize[i] < 100
            columnToNormalize[i] = columnToNormalize[i] * 60000
        end
    end

    return columnToNormalize
end