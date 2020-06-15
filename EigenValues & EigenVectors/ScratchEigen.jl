using LinearAlgebra, Statistics
using DataFrames, RDatasets, DataFramesMeta, CategoricalArrays, Query
using GLM
using Plots

function EigenVector(A; max_iter = 500)
    Q, R = qr(A)
    for iter in 1:max_iter
        A1 = R*Q
        Q, R = qr(A1)
    end
    
    eig_val = diag(R)
    eig_vec = Q
    
    return eig_val, eig_vec
end

@show A = [1 2 3; 4 5 6; 1 1 1]
@show val, vec = EigenVector(A);