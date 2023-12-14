using LinearAlgebra
using Polynomials

# import some convenience operations from DSP
using DSP: ZeroPoleGain,filtfilt,conv

abstract type AbstractFilter end

function bandpass(
        method::FT,
        y::AbstractVector{YT},
        ωc::Tuple{Float64,Float64}
    ) where {FT<:AbstractFilter,YT<:Real}

    ωl,ωh = sort(collect(ωc))
    lowpass_matrix  = lowpass(method,y,ωh)
    highpass_matrix = highpass(method,y,ωl)

    return lowpass_matrix*highpass_matrix*y
end


# creates an approximate lowpass butterworth filter
struct Butterworth <: AbstractFilter
    n::Int
end

function lowpass(
        proto::Butterworth,
        y::AbstractVector{YT},
        ωc::Float64
    ) where YT <: Real
    ϕ = @. (-1)^(0:proto.n) * binomial(proto.n,0:proto.n)
    λ = tan(ωc/2)^(-2*proto.n)
    
    T = length(y)
    Q = zeros(T,T-proto.n)
    
    for t in 1:T-proto.n
        Q[t:(t+proto.n),t] .= ϕ
    end

    Γh = Q*Q'
    Γl = abs.(Γh)
    return (Γl+λ*Γh) \ Γl
end

function highpass(
        proto::Butterworth,
        y::AbstractVector{YT},
        ωc::Float64
    ) where YT <: Real
    ϕ = @. (-1)^(0:proto.n) * binomial(proto.n,0:proto.n)
    λ = tan(ωc/2)^(-2*proto.n)
    
    T = length(y)
    Q = zeros(T,T-proto.n)
    
    for t in 1:T-proto.n
        Q[t:(t+proto.n),t] .= ϕ
    end

    Γh = Q*Q'
    Γl = abs.(Γh)
    return (Γl+λ*Γh) \ (λ*Γh)
end

function frequency_response(proto::Butterworth,ωc::Float64)
    λ = (tan(ωc/2))^(-2*proto.n)
    return ω -> inv(1+λ*(tan(ω/2))^(2*proto.n))
end

struct Henderson <: AbstractFilter
    n::Int
end

function lowpass(
        proto::Henderson,
        y::AbstractVector{YT},
        ωc::Float64
    ) where YT <: Real
    ϕ = @. (-1)^(0:proto.n) * binomial(proto.n,0:proto.n)
    λ = (2*sin(ωc/2))^(-2*proto.n)
    
    T = length(y)
    Q = zeros(T,T-proto.n)
    
    for t in 1:T-proto.n
        Q[t:(t+proto.n),t] .= ϕ
    end

    return (I+λ*Q*Q') \ I
end

function highpass(
        proto::Henderson,
        y::AbstractVector{YT},
        ωc::Float64
    ) where YT <: Real
    ϕ = @. (-1)^(0:proto.n) * binomial(proto.n,0:proto.n)
    λ = (2*sin(ωc/2))^(-2*proto.n)
    
    T = length(y)
    Q = zeros(T,T-proto.n)
    
    for t in 1:T-proto.n
        Q[t:(t+proto.n),t] .= ϕ
    end

    return (I+λ*Q*Q') \ (λ*Q*Q')
end

function frequency_response(proto::Henderson,ωc::Float64)
    λ = (sin(ωc/2))^(-2*proto.n)
    return ω -> inv(1+λ*(sin(ω/2))^(2*proto.n))
end

## CONVOLUTIONAL FILTERS ######################################################

# construct the prototype butterworth filter
function prototype(::Type{T},n::Int64) where {T<:Real}
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

prototype(n::Integer) = prototype(Float64,n)

function bilinear(s::Vector{T}) where T <: Number
    z = similar(s)
    σ = bilinear!(z,s)

    return z,σ
end

# this calculation is twofold: bilinear transform which outputs the DC gain
function bilinear!(z::Vector{T},s::Vector{T}) where T <: Number
    σ = one(1-one(T))
    for i in 1:length(s)
        z[i] = (1+s[i])/(1-s[i])
        σ   *= (1-s[i])
    end

    return real(σ)
end

function semibilinear(s::Vector{T}) where T <: Number
    z = similar(s)
    for i in eachindex(s)
        ω = 2*asin.(-im*s[i])
        z[i] = cis(ω)
    end

    return z
end

# define a lowpass butterworth filter
function lowpass(filter_type::Butterworth,ωc::Float64)
    # define a prototype filter and prewarp the cutoff frequency
    proto = prototype(filter_type.n)
    Ωc = tan(ωc/2)

    # construct the analog lowpass filter
    φ = ZeroPoleGain{:s}(ComplexF64[],Ωc*proto.p,1.0)

    # preallocate the poles and zeros
    z = fill(convert(ComplexF64,-1),length(φ.p))
    p = Vector{ComplexF64}(undef,length(φ.p))

    # bilinear transform them into the discrete time domian
    bilinear!(z,φ.z)
    bilinear!(p,φ.p)

    # normalize the gain
    num = expand_polynomial(p)'*ones(filter_type.n+1)
    den = expand_polynomial(z)'*ones(filter_type.n+1)

    return ZeroPoleGain{:z}(z,p,num/den)
end

# this is a work in progress and has a lot of redundancies I can clean up
function bandpass(filter_type::Butterworth,ωc::Tuple{Float64,Float64})
    # define a prototype filter and prewarp the cutoff frequencies
    ω1,ω2 = tan.(sort(collect(ωc))./2)
    proto = prototype(filter_type.n)

    z = proto.z
    p = proto.p
    k = proto.k

    newz = zeros(ComplexF64,filter_type.n)
    newp = zeros(ComplexF64,2*filter_type.n)
    
    # frequency transform the prototype lowpass to a bandpass filter
    for (oldc,newc) in ((p,newp),(z,newz))
        for i in eachindex(oldc)
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
        oftype(k,k*(ω2-ω1)^(filter_type.n))
    )

    # preallocate the poles and zeros
    z = fill(convert(ComplexF64,-1),length(φ.p))
    p = Vector{ComplexF64}(undef,length(φ.p))

    # bilinear transform them into the discrete time domian
    num = bilinear!(z,φ.z)
    den = bilinear!(p,φ.p)

    return ZeroPoleGain{:z}(z,p,φ.k*(num/den))
end

# define a lowpass henderson filter (not 100% sure, but it's really close)
function lowpass(filter_type::Henderson,ωc::Float64)
    # define a prototype filter and prewarp the cutoff frequency
    proto = prototype(filter_type.n)
    Ωc = sin(ωc/2)

    # construct the analog lowpass filter
    φ = ZeroPoleGain{:s}(ComplexF64[],Ωc*proto.p,1.0)
    
    #z = semibilinear(φ.z)    
    p = semibilinear(φ.p)

    num = expand_polynomial(p)'*ones(filter_type.n+1)
    den = 1

    # then into the digital time domain
    return ZeroPoleGain{:z}(
        ComplexF64[],
        p,
        num/den
    )
end

# while this method works, it is by no means final
function bandpass(filter_type::Henderson,ωc::Tuple{Float64,Float64})
    # define a prototype filter and prewarp the cutoff frequencies
    ω1,ω2 = sin.(sort(collect(ωc))./2)
    proto = prototype(filter_type.n)

    newz = zeros(ComplexF64,filter_type.n)
    newp = zeros(ComplexF64,2*filter_type.n)

    # frequency transform the prototype lowpass to a bandpass filter
    for (oldc,newc) in ((proto.p,newp),(proto.z,newz))
        for i in eachindex(oldc)
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
        oftype(proto.k,proto.k*(ω2-ω1)^(filter_type.n))
    )

    # preallocate the poles and zeros
    z = fill(convert(ComplexF64,-1),length(φ.p))

    # preallocate the poles and zeros
    z[1:filter_type.n] = semibilinear(φ.z)
    p = semibilinear(φ.p)

    # figure out how to get rid of this...
    _,num = bilinear(φ.z)
    _,den = bilinear(φ.p)

    return ZeroPoleGain{:z}(z,p,φ.k*(num/den))
end

## POLYNOMIAL FUNCTIONS #######################################################

function biquadratic(x::AbstractVector{<:Number})
    roots = [[1,-xi] for xi in x]
    
    biquads = Vector{Float64}[]
    for i in 1:div(lastindex(roots),2)
        biquad = real.(conv(roots[(2*i-1):(2*i)]...))
        biquad[1] = 1.0
        push!(biquads,biquad)
    end

    if isodd(lastindex(roots))
        push!(biquads,real(last(roots)))
    end

    return biquads
end

# not as fully featured as DSP, but it works for the relevant IIR filters
function second_order_sections(zpk::ZeroPoleGain{D}) where D
    z = zpk.z
    for i in 1:div(lastindex(z),2)
        z[[2*i-1,2*i]] = sort(z[[2*i-1,2*i]],by=x->imag(x))
    end

    p = sort(zpk.p,by=x->abs(abs(x)-1))
    for i in 1:div(lastindex(p),2)
        p[[2*i-1,2*i]] = sort(p[[2*i-1,2*i]],by=x->imag(x))
    end

    # convert to sequence of biquadratics
    p_biquads = biquadratic(p)
    z_biquads = !isempty(z) ? biquadratic(z) : fill([1.0],length(p_biquads))
    gain = zpk.k^(1/length(p_biquads))

    return (
        b = gain*Polynomial.(z_biquads,D),
        a = Polynomial.(p_biquads,D)
    )
end

# use a convolution over the set of roots to expand the polynomial
# (I could also just use Polynomials.jl)
function expand_polynomial(z::Vector{ZT}) where ZT <: Number
    if isempty(z)
        return 1
    else
        ϕ = [[1,-zi] for zi in z]
        return real(reduce(conv,ϕ))
    end
end

# this could be modified to interface with Polynomials
function polynomial(zpk::ZeroPoleGain{D}) where D
    return (
        b = Polynomial.(zpk.k*expand_polynomial(zpk.z),D),
        a = Polynomial.(expand_polynomial(zpk.p),D)
    )
end

## PLOTTING FUNCTIONS #########################################################

function plot_poles(zpk::ZeroPoleGain)
    real_poles = [(real(p),imag(p)) for p in zpk.p]
    angles = sort([π/2+acos(imag(p)) for p in zpk.p])

    fig = Figure()
    ax = Axis(fig[1,1],aspect=1)

    scatter!(ax,real_poles,color=:black)
    arc!(ax,Point2f(0),1,-π,π,color=:black,linewidth=1)
    arc!(ax,Point2f(0),1.1,angles[1:2]...,color=:black,linestyle=:dot)
    
    return fig
end