include("./dependencies.jl")
include("./utils.jl")
include("./models.jl")
include("./iterators.jl")

cd(output_dir)
results_file_dir = string(output_dir, "/edsr_scale", scale, "_", model_type, "_results.jld2")

r = Knet.load(results_file_dir,"results")

plot([r[1,:], r[2,:]],
     labels=[:trnEDSR :tstEDSR],xlabel="Epochs",ylabel="Loss", title = string("EDSR ", model_type, " Model-Scale ", scale));
savefig(string(result_dir, "loss.png"))""

plot([r[3,:], r[4,:]],
     labels=[:trnEDSR :tstEDSR],xlabel="Epochs",ylabel="PSNR", legend=:bottomright, title = string("EDSR ", model_type, " Model-Scale ", scale))
savefig(string(result_dir, "PSNR.png"))

ispath(string(result_dir, scale, "/", model_type)) || mkdir(string(result_dir, scale, "/", model_type))
cd(string(result_dir, scale, "/", model_type))

open(string("edsr_scale", scale,"_", model_type,"_loss_psnr.txt","w") do io
    write(io, string("EDSR Train Results for ", model_type, " Model Scale ", scale, "\n"))
    for i in 1:length(r[1,:])
        write(io, "Iteration: $i --Loss: $(r[1,:][i]) --PSNR: $(r[3,:][i])\n")
    end
    write(io, "\n\n--------------------------------------\n\n")
    write(io, string("EDSR Test Results for ", model_type, " Model Scale ", scale, "\n"))
    for i in 1:length(r[2,:])
        write(io, "Iteration: $i --Loss: $(r[2,:][i]) --PSNR: $(r[4,:][i])\n")
    end
end

model_file_dir = string(output_dir, "/edsr_scale", scale, "_", model_type, "_model.jld2")

edsrmodel = Knet.load(model_file_dir, string("edsr_scale_", scale, "_", model_type))

tst_dir_lr= string(datadir, "/DIV2K_test_LR_bicubic/X", scale, "/")
tst_dir_hr= string(datadir, "/DIV2K_test_HR")

tst_img_lr= get_dir(tst_dir_lr);
tst_img_hr= get_dir(tst_dir_hr);

test_images_lr = load_images(tst_dir_lr, tst_img_lr)
test_images_hr = load_images(tst_dir_hr, tst_img_hr)

tst_img_clt_lr = collect(test_images_lr);

tst_img_clt_hr = collect(test_images_hr);

img_save(string(result_dir, scale, "/", model_type), tst_img_clt_lr, edsrmodel, scale)

println("Done... Super-resolution images are generated!")


