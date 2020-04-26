include("./utils.jl")

mutable struct ConvModelRelu; w; b; f; end
ConvModelRelu(w1,w2,cx,cy,f=relu) = ConvModelRelu(param(w1,w2,cx,cy), param0(1,1,cy,1), f)
(c::ConvModelRelu)(x) = c.f.(conv4(c.w, x; padding=(1, 1)) .+ c.b)

mutable struct ConvModel; w; b; end
ConvModel(w1,w2,cx,cy) = ConvModel(param(w1,w2,cx,cy), param0(1,1,cy,1))
(c::ConvModel)(x) = (conv4(c.w, x; padding=(1, 1)) .+ c.b)

mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

mutable struct ResBlock3; chainModel; _s; ResBlock3(chainModel,_s) = new(chainModel,_s); end
ResBlock3(w1,w2,cx,cy,s=1.0,f=relu) = begin
                          rb3 = ResBlock3(Chain(
                          ConvModelRelu(param(w1,w2,cx,cy), param0(1,1,cy,1), f),
                          ConvModel(param(w1,w2,cx,cy), param0(1,1,cy,1))),s)
                          rb3
                          end
(rb::ResBlock3)(x) = rb._s * rb.chainModel(x) .+ x

mutable struct ChainResBlock2; resblockChain::Chain; lastConvLayer; ChainResBlock2(rbc, lcv) = new(rbc,lcv); end
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

mutable struct UpsampleLayer; cm::ConvModel; UpsampleLayer(cm) = new(cm); end
UpsampleLayer(w1,w2,cx,cy) = UpsampleLayer(ConvModel(w1,w2,cx,cy))
(usp::UpsampleLayer)(x) = begin
                              convx = usp.cm(x)
                              scx = pixelShuffle2(convx)
                              scx
                          end


mutable struct EDSRModel; cmfirst::ConvModel; crb::ChainResBlock2; ul::UpsampleLayer; cmlast::ConvModel; 
EDSRModel(cmfirst, crb, ul, cmlast)= new(cmfirst, crb, ul, cmlast); end

EDSRModel(w1,w2,cx,cy, n_iter,s=1.0,f=relu) = EDSRModel(ConvModel(w1,w2,cx,cy), ChainResBlock2(w1,w2,cy,cy,n_iter),UpsampleLayer(w1,w2,cy,cy*4),ConvModel(w1,w2,cy,cx))

(edsrm::EDSRModel)(x) = edsrm.cmlast(edsrm.ul(edsrm.crb(edsrm.cmfirst(x))))

mutable struct EDSRModelMean; edsrm::EDSRModel; mean_hc; EDSRModelMean(mean_hc)= new(mean_hc); 
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
