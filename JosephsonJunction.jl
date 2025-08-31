# include("JosephsonJunction.jl")
using Plots
using Unitful
using DifferentialEquations
using PhysicalConstants.CODATA2022
using Statistics
using DataFrames
using CSV
using DynamicalSystems
using ProgressMeter
using DelimitedFiles
using DSP
using LaTeXStrings

const ħ = ReducedPlanckConstant
const e = ElementaryCharge
const k_B = BoltzmannConstant

#####################################################
# Handy functions
#####################################################

function rcsj_equation!(du, u, p, τ)
  ϕ, ω = u
  β, drive_func = p

  i_drive = drive_func(τ,p)

  du[1] = ω
  du[2] = i_drive - sin(ϕ) - ω * β
  
  nothing
end

function rcsj_noise!(du, u, p, τ)

  β, _, Γ = p

  noise_amp = sqrt(2 * β * Γ)

  du[1] = 0.0
  du[2] = noise_amp
  
end

function run_chaos_sim(p, tau_end)
  prob = ODEProblem(rcsj_equation!, [0.0,0.0],[0.0,tau_end],p)
  sol = solve(prob, AutoVern9(Rodas5P()), abstol = 1e-8, reltol = 1e-8)

  return sol
end

function run_noise_chaos_sim(p)
  prob = SDEProblem(rcsj_equation!, rcsj_noise!, [0.0,0.0],[0.0,p.τ_end],p)
  sol = solve(prob, SRIW1())
  return sol
end

function gen_nois_prob(p, tau_end)
  prob = SDEProblem(rcsj_equation!, rcsj_noise!, [0,0], [0.0, tau_end], p)
  
  condition(u, t, integrator) = u[1] - pi / 2
  affect!(integrator) = terminate!(integrator)
  cb = ContinuousCallback(condition, affect!)

  return (prob, cb)
end

##

function normalize_josephson(p)
  β = 1 / sqrt(2 * e / ħ * p.I_c * p.R * p.R * p.C)
  ω_p = sqrt(2 * e * p.I_c / (ħ * p.C))
  i_drive(τ,p_i) = upreferred(p.I_drive(τ / ω_p, p_i) / p.I_c)

  (β = upreferred(β), ω_p = upreferred(ω_p), i_drive = i_drive)
end

function get_noise_gamma(p)
  upreferred((2 * e * k_B * p.T) / (ħ * p.I_c))
end

#####################################################################
# Tools Above, Main codes below
#####################################################################

dIdt = 1e6u"nA/ms"

function I_drive(t,p)
  dIdt * t
end

function run_with_T(temp)
  println(temp)
  jj_par = (R= 0.44u"kΩ", I_c = 2.2u"μA", T = temp, C=0.33u"pF", I_drive = I_drive)

  normal_j = normalize_josephson(jj_par)
  Γ = get_noise_gamma(jj_par)

  println(normal_j)
  println(Γ)
  τ_end = upreferred(normal_j.ω_p * 1.2 * jj_par.I_c / dIdt)
  println(τ_end)

  p = (β = normal_j.β, i_drive = normal_j.i_drive, Γ = Γ)
  prob, cb = gen_nois_prob(p, τ_end)

  function prob_func(prob, i, repeat)
    remake(prob)
  end

  ensemble_prob = EnsembleProblem(prob)
  sim = solve(ensemble_prob, SRIW1(), EnsembleThreads(), 
              trajectories=1000, maxiters=1e9, dt=1,
              callback=cb,
              save_everystep=false, 
              save_start=false)

  ts = map(x -> (x.t[end] / normal_j.ω_p), sim.u)
  Is = map(x -> ustrip(u"μA", I_drive(x, p)), ts)
  println(Is)
  return Is
end


function main_Isw()

Ts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] .* 100.0u"mK"
Iss = map(run_with_T, Ts)
Isw = map(mean, Iss)
Isw_std = map(std, Iss)

open("out.csv", "w") do f
  println(f, "T, Isw, Isw_std")
  for i in eachindex(Ts)
    println(f, join([ustrip(u"K",Ts[i]), Isw[i], Isw_std[i]], ", "))
  end
end

return Iss

end

function i_drive(τ, p)
  return p.amp * sin(p.ω_s * τ)
end

function i_drive_m(τ, p)
  if τ < 2500 || τ > 5000
    return p.amp * sin(p.ω_s * τ)
  end
  return (p.amp + p.add) * sin(p.ω_s * τ)
end

function main_chaos_noise()

  for amp in [0.8, 1.0636, 1.0652, 1.08, 1.212, 1.32]
    p = (β=0.5, i_drive=i_drive, Γ=0.001, amp=amp, ω_s = 0.66, τ_end=800)
    sol = run_noise_chaos_sim(p)
    df = DataFrame(sol)
    CSV.write(join(("chaos_noise_", amp, ".csv"), ""), df)
  end
end

function main_chaos()
  for amp in [0.8, 1.0636, 1.0652, 1.08, 1.212, 1.32]
    p = (β=0.5, i_drive=i_drive, Γ=0.000, amp=amp, ω_s = 0.66, τ_end=800)
    sol = run_chaos_sim(p,p.τ_end)
    df = DataFrame(sol)
    CSV.write(join(("chaos_", amp, ".csv"), ""), df)
  end
end

function main_lle()
  # jj_par = (R= 0.44u"kΩ", I_c = 2.2u"μA", C=0.33u"pF", I_drive = I_drive)
  # normal_j = normalize_josephson(jj_par)
  p_real=(β=0.37, ω_p=22u"GHz", C = 31.8u"pF", R = 615u"Ω", T = 50u"mK", I_c = 200u"nA")

  function get_lle(amp, ω_s)
    p = (β=0.37, i_drive=i_drive, Γ=0.000, amp=amp, ω_s = ω_s)
    ds = CoupledODEs(rcsj_equation!, [0.0,0.0], p, diffeq = (alg = AutoVern9(Rodas5P()), reltol = 1e-6))
    lle = lyapunov(ds, 4000.0, Ttr=2000.0)
    return lle
  end

  # amps = range(0.6, 1.6, length = 200)
  amps = range(0.8, 1.2, length = 200)
  # ω_ss = 10 .^ range(log10(0.1), log10(1.0), 200)
  ω_ss = range(0.4, 0.8, 200)
  # println(ω_ss)
  lles = zeros(length(amps), length(ω_ss))

  indices = CartesianIndices(lles)
  @showprogress Threads.@threads for I in indices
    i, j = I.I
    lle = get_lle(amps[i], ω_ss[j])
    lles[i, j] = lle
  end

  writedlm("lle_matrix.csv", lles, ", ")
  writedlm("lle_omegas.csv", ω_ss, ", ")
  writedlm("lle_amps.csv", amps, ", ")

  # ENV["QT_QPA_PLATFORM"]="xcb"
  gr()
  p = heatmap(ω_ss, amps, max.(lles, 0),
              # xaxis = :log10,
              xlabel = L"$\omega_{s}$", ylabel = L"$i_{a}$",
              tick_direction = :out,
              # xticks = ([0.1, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.1", "0.2", "0.4", "0.6", "0.8", "1.0"]),
              yticks = ([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], ["0.6", "0.8", "1.0", "1.2", "1.4", "1.6"]),
              grid = true, minorgird = true,
              framestyle = :box,
              colorbar_title = L"LLE",)
  savefig(p, "lle_heat.pdf")
  display(p)

  return lles

end

function show_detection()
  p = (β=0.37, i_drive=i_drive_m, Γ=0.000, amp=1.07, ω_s = 0.6773, τ_end=750, add = 0.08)
  sol = run_noise_chaos_sim(p)
  ENV["QT_QPA_PLATFORM"]="xcb"
  gr()
  plot(sol, vars=(2))
end

# function get_LLE_from_sol(sol, )