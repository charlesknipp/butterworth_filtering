using LinearAlgebra
using DSP
using Plots

## HODRICK PRESCOTT MODEL #####################################################

# TODO: rely less on DSP.jl, should only use conv()

# construct the prototype butterworth filter
function BFT(::Type{T},n::Int64) where {T<:Real}
    n > 0 || error("n must be positive")
    poles = zeros(Complex{T},n)

    # this is stripped straight from DSP.jl
    for i in 1:div(n,2)
        θ = convert(T,2*i-1)/(2*n)
        pole = complex(-sinpi(θ),cospi(θ))
        poles[2*i-1] = pole
        poles[2*i] = conj(pole)
    end

    # for odd order numbers, set the trailing pole to -1
    if isodd(n)
        poles[end] = -1
    end

    return ZeroPoleGain{:s}(Complex{T}[],poles,one(T))
end

BFT(n::Integer) = BFT(Float64,n)


function bilinear(s::Vector{T}) where T <: Number
    z = similar(s)
    σ = bilinear!(z,s)

    return z,σ
end

function bilinear!(z::Vector{T},s::Vector{T}) where T <: Number
    σ = one(1-one(T))

    # this calculation is twofold: bilinear transform which outputs the DC gain
    for i in 1:length(s)
        z[i] = (1+s[i])/(1-s[i])
        σ   *= (1-s[i])
    end

    return real(σ)
end

# define a lowpass butterworth filter
function BFT(ωc::Float64;n::Int64=2)
    # define a prototype filter and prewarp the cutoff frequency
    proto = BFT(n)
    λ = tan(ωc/2)

    # construct the analog lowpass filter
    φ = ZeroPoleGain{:s}(
        λ*proto.z,
        λ*proto.p,
        proto.k*λ^(length(proto.p)-length(proto.z))
    )

    # preallocate the poles and zeros
    z = fill(convert(ComplexF64,-1),length(φ.p))
    p = Vector{ComplexF64}(undef,length(φ.p))

    # bilinear transform them into the discrete time domian
    num = bilinear!(z,φ.z)
    den = bilinear!(p,φ.p)

    return ZeroPoleGain{:z}(z,p,φ.k*(num/den))
end

# this is a work in progress and has a lot of redundancies I can clean up
function BFT(ωc::Tuple{Float64,Float64};n::Int64=2)
    # define a prototype filter and prewarp the cutoff frequencies
    ω1,ω2 = tan.(sort(collect(ωc))./2)
    proto = BFT(n)

    z = proto.z
    p = proto.p
    k = proto.k

    newz = zeros(ComplexF64,n)
    newp = zeros(ComplexF64,2*n)
    
    # frequency transform the prototype lowpass to a bandpass filter
    for (oldc,newc) in ((p,newp),(z,newz))
        for i = 1:length(oldc)
            b  = oldc[i]*((ω2-ω1)/2)
            pm = sqrt(b^2-ω2*ω1)

            newc[2*i-1] = b + pm
            newc[2*i]   = b - pm
        end
    end

    # construct the analog bandpass filter
    φ = ZeroPoleGain{:s}(
        newz,
        newp,
        oftype(k,k*(ω2-ω1)^(n))
    )

    # preallocate the poles and zeros
    z = fill(convert(ComplexF64,-1),length(φ.p))
    p = Vector{ComplexF64}(undef,length(φ.p))

    # bilinear transform them into the discrete time domian
    num = bilinear!(z,φ.z)
    den = bilinear!(p,φ.p)

    return ZeroPoleGain{:z}(z,p,φ.k*(num/den))
end

function expand_polynomial(z::Vector{ZT}) where ZT <: Number
    # let z represent the zeros of the polynomial with real coefficients a
    a = [1,-z[1]]

    # this convolution could totally be wrong, but idk
    for i in 2:length(z)
        a = conv(a,[1,-z[i]])
    end

    # coefficients must be real
    return real(a)
end

# test these functions
zpk = BFT(π/32,n=2)
zpk = BFT((π/6,π/32),n=2)

# polynomial expansion
a = expand_polynomial(zpk.p)
b = (zpk.k)*expand_polynomial(zpk.z)

# compare to DSP
convert(
    PolynomialRatio,
    digitalfilter(
        Bandpass(π/32,π/6,fs=2π),
        Butterworth(2)
    )
)
