include("butterworth.jl")
using FredData

# ENV["https_proxy"] = "http://wwwproxy.frb.gov:8080"

# consumption expenditures data
inflation_data = get_data(
    Fred(),
    "PCECTPI",
    observation_start = "1947-04-01",
    units = "cca",
    frequency = "q"
)

# construct final dataset
data = select(
    inflation_data.data,
    :date,
    :value
)

###############################################################################

function compare_methods(filter_type::FT,data,ωc::Float64) where FT<:AbstractFilter
    ridge_reg = lowpass(filter_type,data,ωc)*data
    conv_filt = filtfilt(lowpass(filter_type,ωc),data)

    fig = Figure()
    ax = Axis(fig[1,1],limits=(nothing,(-10,15)))

    lines!(ax,data,linewidth=2,linestyle=:dot,color=:black,alpha=0.5)
    lines!(ax,conv_filt,linewidth=2,color=:red,label="convolution")
    lines!(ax,ridge_reg,linewidth=2,color=:blue,label="ridge regression")

    Legend(fig[2,1],ax,orientation=:horizontal,nbanks=2)

    return fig
end

function compare_methods(filter_type::FT,data,ωc::Tuple{Float64,Float64}) where FT<:AbstractFilter
    ridge_reg = bandpass(filter_type,data,ωc)
    conv_filt = filtfilt(bandpass(filter_type,ωc),data)
    noisy_cycle = data-filtfilt(lowpass(filter_type,min(ωc...)),data)

    fig = Figure()
    ax = Axis(fig[1,1],limits=(nothing,(-7,7)))

    lines!(ax,noisy_cycle,linewidth=2,linestyle=:dot,color=:black,alpha=0.5)
    lines!(ax,conv_filt,linewidth=2,color=:red,label="convolution")
    lines!(ax,ridge_reg,linewidth=2,color=:blue,label="ridge regression")

    Legend(fig[2,1],ax,orientation=:horizontal,nbanks=2)

    return fig
end

###############################################################################

# for lowpass filters
compare_methods(Henderson(2),data.value,π/6)
compare_methods(Butterworth(2),data.value,π/6)

# for bandpass filters
compare_methods(Henderson(2),data.value,(π/6,π/32))
compare_methods(Butterworth(2),data.value,(π/6,π/32))
