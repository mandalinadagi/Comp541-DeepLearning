
ENV["COLUMNS"]=72
using Pkg; for p in ("Plots", "FileIO", "IterTools", "Images", "Colors", "TestImages", "Knet", "ImageIO", "Random"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Base.Iterators: flatten
using IterTools: ncycle, takenth
using Plots
using Images, Knet, Random, ImageIO, TestImages, Colors
using Statistics: mean

println("Dependencies are registered...")
