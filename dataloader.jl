include("utils.jl")

trn_dir_lr= string(datadir, "/DIV2K_train_LR_bicubic/X", scale)
trn_dir_hr= string(datadir, "/DIV2K_train_HR")
tst_dir_lr= string(datadir, "/DIV2K_test_LR_bicubic/X", scale)
tst_dir_hr= string(datadir, "/DIV2K_test_HR")


trn_img_lr= get_dir(trn_dir_lr);
trn_img_hr= get_dir(trn_dir_hr);
tst_img_lr= get_dir(tst_dir_lr);
tst_img_hr= get_dir(tst_dir_hr);

train_images_lr = load_images(trn_dir_lr, trn_img_lr)
train_images_hr = load_images(trn_dir_hr, trn_img_hr)
test_images_lr = load_images(tst_dir_lr, tst_img_lr)
test_images_hr = load_images(tst_dir_hr, tst_img_hr)

trn_img_clt_lr = collect(train_images_lr);
trn_img_clt_hr = collect(train_images_hr);

tst_img_clt_lr = collect(test_images_lr);
tst_img_clt_hr = collect(test_images_hr);

println("Done. Data loaded successfully!")

dtrn = DData2(Minb4(shuffle_img(trn_img_clt_lr, trn_img_clt_hr)..., patchsize, scale, minibatchsize));

dtst = DData2(DataTest(tst_img_clt_lr, tst_img_clt_hr));

println("Train and Test data are ready!")

cd("../../edsr")
