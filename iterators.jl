include("./utils.jl")

mutable struct Minb4; 
xs; ys; patchsize; scale; minibatchsize; 
Minb4(xs, ys, patchsize, scale, minibatchsize)=new(xs, ys, patchsize, scale, minibatchsize)
end

function Base.iterate(iter::Minb4, state=0)
    if(state==length(iter))
        return nothing
    end
    
    patches_x = Any[]
    patches_y = Any[]
    #s = (state)%(length(iter))
    s = state
    for j in 1:iter.minibatchsize
        patch_pair = takepatch([iter.xs[s*16 + j],iter.ys[s*16 + j]], iter.patchsize, iter.scale)
        push!(patches_x, patch_pair[1])
        push!(patches_y, patch_pair[2])
    end
        
        return [cat(patches_x...,dims=4),cat(patches_y...,dims=4)] , s+1
end

function Base.length(iter::Minb4)
    Int(length(iter.xs)/iter.minibatchsize)
end

mutable struct DataTest; xs; ys;
DataTest(xs, ys)= new(xs, ys); end

function Base.iterate(itertest::DataTest, state=0)
     if(state==length(itertest))
        return nothing
    end
    println(state)
    s = state+1
    xtst = KnetArray(rgb_images(itertest.xs[s]))
    ytst = KnetArray(rgb_images(itertest.ys[s]))
        
        return [add_dim(xtst),add_dim(ytst)] , s
end

function Base.length(itertest::DataTest)
    Int(length(itertest.xs))
end

mutable struct DData2; d; DData2(d) = new(d); end;

function Base.iterate(f::DData2, s...)
    next = iterate(f.d, s...)
    next === nothing && return nothing
    ((x,y),state) = next
    return (x,y), state
end

Base.length(f::DData2) = length(f.d) # collect needs this

mutable struct Decay_LR3 ; curriter ; itr ; model ; decay; n_batches; end


decay_LR(itr, model, decay, n_batches) = Decay_LR3(0,itr,model,decay, n_batches)

function Base.iterate(f::Decay_LR3, s...)
    next = iterate(f.itr, s...)
    next === nothing && return nothing
    (x,state) = next
    f.curriter += 1
    if(f.curriter == f.n_batches)
        decay_lr(f.model,decay)
    end
    
    #println(params(f.model)[1].opt.lr)

    
    return x , state
end

Base.length(f::Decay_LR3) = length(f.itr)
