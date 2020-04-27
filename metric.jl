
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

println("Evaluation metric: PSNR is generated.")
