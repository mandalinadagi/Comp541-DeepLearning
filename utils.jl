function get_dir(directory)
cd(directory)

array_dir= Any[]
for i in walkdir(directory)
    push!(array_dir, i)
end
array_dir = array_dir[1][3]
end

function rgb_images(img)
    Float32.(cat(broadcast(red,img),broadcast(green,img),broadcast(blue,img), dims=3))
    
end

function load_images(directory, array_dir)
    (load(string(directory ,"/",x)) for x in array_dir)
end

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

add_dim(x::Array) = reshape(x, (size(x)...,1))
add_dim(x::KnetArray) = reshape(x, (size(x)...,1))

function shuffle_img(clt_lr, clt_hr)
    ln = length(clt_lr)
    a = randperm(ln)
    clt_lr_new = clt_lr[a]
    clt_hr_new = clt_hr[a]
    
    return [clt_lr_new, clt_hr_new]
    
end

function pixelShuffle2(x)
  xsize = size(x)
  
  xr = reshape(x, (xsize[1],xsize[2],2,2,convert(Int,xsize[3]/4),xsize[4]))
  xrp = permutedims(xr, (3,1,4,2,5,6))
  xrpr = reshape(xrp, xsize[1]*2, xsize[2]*2,convert(Int,xsize[3]/4),xsize[4])

  xrpr
end

decay = 0.5;

function decay_lr(model, decay)
    for par in params(model)
        par.opt.lr *= decay
    end
end

function test_sr(model, image)
    img_show= clamp.(dropdims(Array(model(image)), dims=4), 0, 1)
    img_color= colorview(RGB, img_show[:,:,1], img_show[:,:,2], img_show[:,:,3])
end

function img_save(dir, tst_img_clt_lr, model, scale)
    for i in 1:length(tst_img_clt_lr)
       sr =test_sr(model, add_dim(Knet.KnetArray(rgb_images(tst_img_clt_lr[i]))))
       save(string(dir, "SR_X" , scale, "/scale_", scale, "_", i, ".png"), sr) 
    end
end
