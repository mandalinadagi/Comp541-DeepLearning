cd("/kuacc/users/ckorkmaz14/comp541_project/edsr/")
include("./dependencies.jl")
include("./utils.jl")

include("./iterators.jl")
include("./dataloader.jl")
include("./models.jl")
include("./metric.jl")

cd("/kuacc/users/ckorkmaz14/comp541_project/edsr/models")

output_features = 64;
n_iter =16;

function trainresults(file,model; o...)
    epoch_num = 300;
    iter_no_for_decay= 200;
    evaluate_every = 10;
    #if (print("Train from scratch? "); readline()[1]=='y')
        r = ((model(DData2(dtrn)), model(DData2(dtst)), calc_PSNR(model,dtrn), calc_PSNR(model,dtst))
             for x in takenth(progress(decay_LR(adam(model,DData2(ncycle(dtrn,epoch_num)); lr=0.0001),model,decay, iter_no_for_decay*50)),length(dtrn)*evaluate_every))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        Knet.gc() # To save gpu memory
    #else
    #    r = Knet.load(file,"results")
    #end
    #println(minimum(r,dims=2))
    return r
end



Knet.gc()

edsr_scale2_baseline = EDSRModelMean(3,3,3,output_features,n_iter);

trn_260420 = trainresults("scale2_baseline_results.jld2", edsr_scale2_baseline)

Knet.save("scale2_baseline_model.jld2", "edsrm",edsr_scale2_baseline)

println("Model is saved!")


