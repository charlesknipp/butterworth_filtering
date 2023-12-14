include("butterworth.jl")

using CairoMakie
using DataFrames,CSV

pce_data = CSV.read("data/pce.csv",DataFrame)
gdp_data = CSV.read("data/real_gdp.csv",DataFrame)

###############################################################################

function compare_methods(filter_type::FT,data,ωc::Float64;ylims=nothing) where FT<:AbstractFilter
    ridge_reg = lowpass(filter_type,data,ωc)*data
    conv_filt = filtfilt(lowpass(filter_type,ωc),data)

    fig = Figure()
    ax = Axis(fig[1,1],limits=(nothing,ylims))

    lines!(ax,data,linewidth=2,linestyle=:dot,color=:black,alpha=0.5)
    lines!(ax,conv_filt,linewidth=2,color=:red,label="convolution")
    lines!(ax,ridge_reg,linewidth=2,color=:blue,label="ridge regression")

    Legend(fig[2,1],ax,orientation=:horizontal,nbanks=2)

    return fig
end

function compare_methods(filter_type::FT,data,ωc::Tuple{Float64,Float64};ylims=nothing) where FT<:AbstractFilter
    ridge_reg = bandpass(filter_type,data,ωc)
    conv_filt = filtfilt(bandpass(filter_type,ωc),data)
    noisy_cycle = data-filtfilt(lowpass(filter_type,min(ωc...)),data)

    fig = Figure()
    ax = Axis(fig[1,1],limits=(nothing,ylims))

    lines!(ax,noisy_cycle,linewidth=2,linestyle=:dot,color=:black,alpha=0.5)
    lines!(ax,conv_filt,linewidth=2,color=:red,label="convolution")
    lines!(ax,ridge_reg,linewidth=2,color=:blue,label="ridge regression")

    Legend(fig[2,1],ax,orientation=:horizontal,nbanks=2)

    return fig
end

## PCE ########################################################################

# for lowpass filters
compare_methods(Henderson(2),pce_data.value,π/12,ylims=(-10,15))
compare_methods(Butterworth(2),pce_data.value,π/12,ylims=(-10,15))

# for bandpass filters
compare_methods(Henderson(2),pce_data.value,(π/6,π/32),ylims=(-7,7))
compare_methods(Butterworth(2),pce_data.value,(π/6,π/32),ylims=(-7,7))

## GDP ########################################################################

# for lowpass filters
compare_methods(Henderson(2),gdp_data.value,π/12)
compare_methods(Butterworth(2),gdp_data.value,π/12)

# for bandpass filters
compare_methods(Henderson(2),gdp_data.value,(π/6,π/32),ylims=(-7,7))
compare_methods(Butterworth(2),gdp_data.value,(π/6,π/32),ylims=(-7,7))

###############################################################################

bp = bandpass(Henderson(5),(π/6,π/32))
lp = lowpass(Henderson(5),π/6)

sos_bp = second_order_sections(bp)
sos_lp = second_order_sections(lp)