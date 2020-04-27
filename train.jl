import Pkg
Pkg.add("ArgParse")
using ArgParse

function main(args)
    pars = ArgParseSettings()
    pars.description = "EDSR Implementation in Knet"

    @add_arg_table pars begin
        ("--scale"; arg_type=Int; default=2; help="super-resolution scale")
        ("--res_scale"; arg_type=Float64; default=1.0; help="residual scaling")
        ("--model_type"; arg_type=String; default="baseline"; help="type of model one of: [baseline(16ResBlocks,64outputfeature), original(32ResBlocks,256outputfeature)]")
        ("--res_block"; arg_type=Int; default=16; help="number of residual blocks")
        ("--output_features"; arg_type=Int; default=64; help="number of output feature channels")
        ("--datadir"; arg_type=String; default="/kuacc/users/ckorkmaz14/comp541_project/data"; help="dataset directory")
        ("--batchsize"; arg_type=Int; default=16; help="input batch size for training")
        ("--patchsize"; arg_type=Int; default=48; help="input patch size for training")
        ("--lr"; arg_type=Float64; default=0.0001; help="learning rate")
        ("--decay"; arg_type=Float64; default=0.5; help="learning rate decay")
        ("--epochs"; arg_type=Int; default=3; help="number of training epochs")
        ("--evaluate_every"; arg_type=Int; default=1; help="report PSNR in n iterations")
        ("--decay_no"; arg_type=Int; default=2; help="decay learning rate after nth epoch")  
        ("--output_dir"; arg_type=String; default="/kuacc/users/ckorkmaz14/comp541_project/edsr/models"; help="output directory for saving model")
        ("--result_dir"; arg_type=String; default="/kuacc/users/ckorkmaz14/comp541_project/edsr/results"; help="super-resolution image director")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, pars; as_symbols=true)

    global scale = o[:scale]
    global s = o[:res_scale]
    global model_type = o[:model_type]
    global n_iter = o[:res_block]
    global output_features = o[:output_features]
    global datadir = o[:datadir]
    global minibatchsize = o[:batchsize]
    global patchsize = o[:patchsize]
    global lr = o[:lr]
    global decay = o[:decay]
    global num_epochs = o[:epochs]
    global evaluate_every = o[:evaluate_every]
    global iter_no_for_decay= o[:decay_no]
    global output_dir = o[:output_dir]
    global result_dir = o[:result_dir]
   
    println("Dataset directory: $datadir")
    println("Super-resolution scale: $scale")
    println("Model type: $model_type")
    println("Number of residual blocks: $n_iter")
    println("Number of output feature channels: $output_features")
    
    println("Minibatch size: $minibatchsize")
    println("Patch size: $patchsize")
    
    println("Learning rate for Adam optimizer: $lr")
    println("Number of training epochs: $num_epochs")
    
    include("./dependencies.jl")

    include("./utils.jl")

    include("./iterators.jl")

    include("./dataloader.jl")

    include("./models.jl")

    include("./metric.jl")

    ispath(output_dir) || mkdir(output_dir)
    cd(output_dir)

 end

main(ARGS)

function trainresults(file,model; o...)
     r = ((model(DData2(dtrn)), model(DData2(dtst)), calc_PSNR(model,dtrn), calc_PSNR(model,dtst))
             for x in takenth(progress(decay_LR(adam(model,DData2(ncycle(dtrn,num_epochs)); lr=lr),model,decay, iter_no_for_decay*50)),length(dtrn)*evaluate_every))
     r = reshape(collect(Float32,flatten(r)),(4,:))
     Knet.save(file,"results",r)
     Knet.gc()
     println(minimum(r,dims=2))
     return r
end
    
#edsr_scale2_baseline = EDSRModelMean(3,3,3,64,16);
    
trn_result = trainresults(string("edsr_scale",scale,"_", model_type, "_results.jld2"), edsr_scale2_baseline)
    
Knet.save(string("edsr_scale", scale, "_", model_type, "_model.jld2"), string("edsr_scale_", scale, "_", model_type), edsr_scale2_baseline)
    
println("Training is done and model is saved!")

println("Output directory: $output_dir")
