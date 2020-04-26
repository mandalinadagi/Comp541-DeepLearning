
ENV["COLUMNS"]=72
using Pkg; for p in ("Plots", "FileIO", "IterTools", "Images", "Colors", "TestImages", "Knet", "ImageIO", "Random"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Base.Iterators: flatten
using IterTools: ncycle, takenth
using Plots
using Images, Knet, Random, ImageIO, TestImages, Colors
using Statistics: mean
using Knet: Knet, conv4, pool, mat, KnetArray, nll, progress, sgd, dropout, relu, Data, Param, abs
#using ImageView

trn_dir_lr= "/kuacc/users/ckorkmaz14/comp541_project/data/DIV2K_train_LR_bicubic/X2/"
trn_dir_hr= "/kuacc/users/ckorkmaz14/comp541_project/data/DIV2K_train_HR"
tst_dir_lr= "/kuacc/users/ckorkmaz14/comp541_project/data/DIV2K_test_LR_bicubic"
tst_dir_hr= "/kuacc/users/ckorkmaz14/comp541_project/data/DIV2K_test_HR"


function get_dir(directory)
cd(directory)

array_dir= Any[]
for i in walkdir(directory)
    push!(array_dir, i)
end
array_dir = array_dir[1][3]
end

trn_img_lr= get_dir(trn_dir_lr);
trn_img_hr= get_dir(trn_dir_hr);
tst_img_lr= get_dir(tst_dir_lr);
tst_img_hr= get_dir(tst_dir_hr);

function rgb_images(img)
    Float32.(cat(broadcast(red,img),broadcast(green,img),broadcast(blue,img), dims=3))
    
end

function load_images(directory, array_dir)
    (load(string(directory ,"/",x)) for x in array_dir)
end

train_images_lr = load_images(trn_dir_lr, trn_img_lr)
train_images_hr = load_images(trn_dir_hr, trn_img_hr)
test_images_lr = load_images(tst_dir_lr, tst_img_lr)
test_images_hr = load_images(tst_dir_hr, tst_img_hr)

#train_images = zip(train_images_lr, train_images_hr)
#test_images = zip(test_images_lr, test_images_hr)

function takepatch(imgpair, patchsize, scale)
    s1, s2 = size(imgpair[1])
    nx = rand((1:s1-patchsize+1))
    ny = rand((1:s2-patchsize+1))
    
    lr_patch = imgpair[1][nx:nx+patchsize-1, ny:ny+patchsize-1, :]
    hr_patch = imgpair[2][scale*(nx-1)+1:scale*(nx+patchsize-1), scale*(ny-1)+1:scale*(ny+patchsize-1), :]
    
    lr_patch = Knet.KnetArray(rgb_images(lr_patch))
    hr_patch = Knet.KnetArray(rgb_images(hr_patch))
    
    return [lr_patch, hr_patch]
        
    
end

scale =2;
patchsize= 48;
minibatchsize = 16;

function iter_images4(img_dir, patchsize, scale, minibatchsize)
    
    xy_patch1 = (takepatch(imp, patchsize,scale) for imp in img_dir)
    xp1 = (x for (x,y) in xy_patch1)
    yp1 = (y for (x,y) in xy_patch1)
    xp16 = Iterators.partition(xp1, minibatchsize)
    yp16 = Iterators.partition(yp1, minibatchsize)
    catxp16 = (cat(z..., dims=4) for z in xp16)
    catyp16 = (cat(z..., dims=4) for z in yp16)
    zip(catxp16,catyp16)
end
    

struct Minb4; 
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

trn_img_clt_lr = collect(train_images_lr);
trn_img_clt_hr = collect(train_images_hr);

tst_img_clt_lr = collect(test_images_lr);

tst_img_clt_hr = collect(test_images_hr);

struct ConvModelRelu; w; b; f; end
ConvModelRelu(w1,w2,cx,cy,f=relu) = ConvModelRelu(param(w1,w2,cx,cy), param0(1,1,cy,1), f)
(c::ConvModelRelu)(x) = c.f.(conv4(c.w, x; padding=(1, 1)) .+ c.b)

struct ConvModel; w; b; end
ConvModel(w1,w2,cx,cy) = ConvModel(param(w1,w2,cx,cy), param0(1,1,cy,1))
(c::ConvModel)(x) = (conv4(c.w, x; padding=(1, 1)) .+ c.b)


struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)



struct ResBlock3; chainModel; _s; ResBlock3(chainModel,_s) = new(chainModel,_s); end
ResBlock3(w1,w2,cx,cy,s=1.0,f=relu) = begin
                          rb3 = ResBlock3(Chain(
                          ConvModelRelu(param(w1,w2,cx,cy), param0(1,1,cy,1), f),
                          ConvModel(param(w1,w2,cx,cy), param0(1,1,cy,1))),s)
                          #rb3.chainModel = Chain(
                          #ConvModelRelu(param(w1,w2,cx,cy), param0(1,1,cy,1), f),
                          #ConvModel(param(w1,w2,cx,cy), param0(1,1,cy,1)))
                          #_s = s
                          rb3
                          end
(rb::ResBlock3)(x) = rb._s * rb.chainModel(x) .+ x


struct ChainResBlock2; resblockChain::Chain; lastConvLayer; ChainResBlock2(rbc, lcv) = new(rbc,lcv); end
ChainResBlock2(w1,w2,cx,cy,n_iter,s=1.0,f=relu) = begin
                                              resblocks = []
                                                for i in 1:n_iter
                                                push!(resblocks, ResBlock3(w1,w2,cx,cy,s,f))
                                              end
                                              print(n_iter)
                                              mychain = Chain(resblocks...)
                                              crb = ChainResBlock2(mychain,ConvModel(w1,w2,cx,cy))
                                              crb
                                              end
(crb::ChainResBlock2)(x) = crb.lastConvLayer(crb.resblockChain(x)) .+ x

struct UpsampleLayer; cm::ConvModel; UpsampleLayer(cm) = new(cm); end
UpsampleLayer(w1,w2,cx,cy) = UpsampleLayer(ConvModel(w1,w2,cx,cy))
(usp::UpsampleLayer)(x) = begin
                              convx = usp.cm(x)
                              scx = pixelShuffle2(convx)
                              scx
                          end


function pixelShuffle2(x)
  xsize = size(x)
  
  xr = reshape(x, (xsize[1],xsize[2],2,2,convert(Int,xsize[3]/4),xsize[4]))
  xrp = permutedims(xr, (3,1,4,2,5,6))
  xrpr = reshape(xrp, xsize[1]*2, xsize[2]*2,convert(Int,xsize[3]/4),xsize[4])

  xrpr
end

struct DData2; d; DData2(d) = new(d); end;

struct EDSRModel; cmfirst::ConvModel; crb::ChainResBlock2; ul::UpsampleLayer; cmlast::ConvModel; 
EDSRModel(cmfirst, crb, ul, cmlast)= new(cmfirst, crb, ul, cmlast); end

EDSRModel(w1,w2,cx,cy, n_iter,s=1.0,f=relu) = EDSRModel(ConvModel(w1,w2,cx,cy), ChainResBlock2(w1,w2,cy,cy,n_iter),UpsampleLayer(w1,w2,cy,cy*4),ConvModel(w1,w2,cy,cx))

(edsrm::EDSRModel)(x) = edsrm.cmlast(edsrm.ul(edsrm.crb(edsrm.cmfirst(x))))

#(edsrm::EDSRModel)(x,y) = mean(broadcast(abs,edsrm(x)-y))
#(edsrm::EDSRModel)(d::DData2) = mean(edsrm(x,y) for (x,y) in d)

struct EDSRModelMean; edsrm::EDSRModel; mean_hc; EDSRModelMean(mean_hc)= new(mean_hc); 
EDSRModelMean(edsrm, mean_hc)= new(edsrm, mean_hc);
end 

EDSRModelMean(w1,w2,cx,cy, n_iter,s=1.0,f=relu, mhc=(0.4488, 0.4371, 0.4040))= EDSRModelMean(EDSRModel(w1,w2,cx,cy, n_iter), mhc)
    

(edsrmmean::EDSRModelMean)(x,y) = mean(broadcast(abs,edsrmmean(x)-y))
(edsrmmean::EDSRModelMean)(d::DData2) = mean(edsrmmean(x,y) for (x,y) in d)

   
(edsrmmean::EDSRModelMean)(x) = begin    
    sub_x = zeros(Float32, size(x))
    
    sub_x[:,:,1,:]= sub_x[:,:,1,:].+edsrmmean.mean_hc[1]
    sub_x[:,:,2,:]= sub_x[:,:,2,:].+edsrmmean.mean_hc[2]
    sub_x[:,:,3,:]= sub_x[:,:,3,:].+edsrmmean.mean_hc[3]
    

    k_sub_x = KnetArray(sub_x)
    
    x_new = x - k_sub_x
    
    ypred_meansubt = edsrmmean.edsrm(x_new)

    sub_y = zeros(Float32, size(ypred_meansubt))

    sub_y[:,:,1,:]= sub_y[:,:,1,:].+edsrmmean.mean_hc[1]
    sub_y[:,:,2,:]= sub_y[:,:,2,:].+edsrmmean.mean_hc[2]
    sub_y[:,:,3,:]= sub_y[:,:,3,:].+edsrmmean.mean_hc[3]

    k_sub_y = KnetArray(sub_y)


    ret_val = ypred_meansubt + k_sub_y
end

   
# (edsrmmean::EDSRModelMean)(x) = begin    
#     x_per= permutedims(x, 1,2,4,3)
#     x_rs = reshape(x_per, (36864, 3))
#     i = KnetArray(zeros(Float32, 9))
#     i = reshape(i, (3,3))
#     i[1,1]= edsrmmean.mean_hc[1]
#     i[2,2]= edsrmmean.mean_hc[2]
#     i[3,3]= edsrmmean.mean_hc[3]
#     x_new = x_rs * i
#     x_new_rs = reshape(x_per, (48,48,16,3)) 
#     x_new_per = permutedims(x_new_rs, 1,2,4,3)
    
#     edsrmmean.edsrm(x_new_per)
    
# end

function shuffle_img(clt_lr, clt_hr)
    ln = length(clt_lr)
    a = randperm(ln)
    clt_lr_new = clt_lr[a]
    clt_hr_new = clt_hr[a]
    
    return [clt_lr_new, clt_hr_new]
    
end

dtrn = DData2(Minb4(shuffle_img(trn_img_clt_lr, trn_img_clt_hr)..., patchsize, scale, minibatchsize))

dtst = DData2(Minb4(tst_img_clt_lr[1:96], tst_img_clt_hr[1:96], patchsize, scale, minibatchsize))

add_dim(x::Array) = reshape(x, (size(x)...,1))
add_dim(x::KnetArray) = reshape(x, (size(x)...,1))

#for 1 image
#dtrn = DData2([(add_dim(Knet.KnetArray(rgb_images(iterate(trn_img_clt_lr)[1][1:150, 1:150, :]))), add_dim(Knet.KnetArray(rgb_images(iterate(trn_img_clt_hr)[1][1:300,1:300,:]))))])

#for 1 image
#dtst = DData2([(add_dim(Knet.KnetArray(rgb_images(iterate(trn_img_clt_lr)[1][1:150, 1:150, :]))), add_dim(Knet.KnetArray(rgb_images(iterate(trn_img_clt_hr)[1][1:300,1:300,:]))))])

function Base.iterate(f::DData2, s...)
    next = iterate(f.d, s...)
    next === nothing && return nothing
    ((x,y),state) = next
    return (x,y), state
end

Base.length(f::DData2) = length(f.d) # collect needs this

function calc_MSE(y_pred, y_gold)
    mean((y_pred-y_gold).^2)
end

function calc_PSNR(model, data)
    psnr_val = Any[]
    for d in data
        y_pred = Array(model(d[1]))
        y_pred = clamp.(y_pred,0.0,1.0)
        y_pred *= 255
        y_pred = round.(y_pred)
        #y_pred = round.(clamp.(y_pred, 0.0, 1.0) * 255)

        y_gold = d[2] * 255
        mse= calc_MSE(y_pred, Array(y_gold))
        psnr = 10*log10((255^2)/mse)
        push!(psnr_val, psnr)
    end
    mean(psnr_val)
end

function trainresults(file,model; o...)
    if (print("Train from scratch? "); readline()[1]=='y')
        r = ((model(DData2(dtrn)), model(DData2(dtst)), calc_PSNR(model,dtrn), calc_PSNR(model,dtst))
             for x in takenth(progress(adam(model,DData2(ncycle(dtrn,10)); lr=0.0001)),length(dtrn)))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        #Knet.save(file, "model", model)
        Knet.gc() # To save gpu memory
    else
        r = Knet.load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end



Knet.gc()

#edsrm = EDSRModel(3,3,3,64,16);
edsrmmean = EDSRModelMean(3,3,3,64,16);

r_test_130420 = trainresults("first_model", edsrmmean)

# plot([r_test_130420[1,:], r_test_130420[2,:]],
#      labels=[:trnEDSR :tstEDSR],xlabel="Epochs",ylabel="Loss")

# plot([r_test_130420[3,:], r_test_130420[4,:]],
#      labels=[:trnEDSR :tstEDSR],xlabel="Epochs",ylabel="PSNR")

function test_sr(model, image)
    img_show= clamp.(dropdims(Array(model(image)), dims=4), 0, 1)
    img_color= colorview(RGB, img_show[:,:,1], img_show[:,:,2], img_show[:,:,3])
    #imshow(img_show)
end

#lr=rgb_images(iterate(trn_img_clt_lr)[1][1:150,1:150, :])
#colorview(RGB, lr[:,:,1], lr[:,:,2], lr[:, :, 3])

#t =test_sr(edsrm, add_dim(Knet.KnetArray(rgb_images(iterate(trn_img_clt_lr)[1][1:150, 1:150, :]))))


#hr=rgb_images(iterate(trn_img_clt_hr)[1][1:300,1:300, :])
#colorview(RGB, hr[:,:,1], hr[:,:,2], hr[:, :, 3])
