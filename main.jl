include("JosephsonJunction.jl")
using .JosephsonJunction
using Plots
using Unitful
using DifferentialEquations
using Statistics

function main()

dIdt = 0.1e6u"nA/ms"
function I_drive(t)
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

  prob, cb = gen_nois_prob((normal_j.β, normal_j.i_drive, Γ), τ_end)

  function prob_func(prob, i, repeat)
    remake(prob)
  end

  ensemble_prob = EnsembleProblem(prob)
  sim = solve(ensemble_prob, SRA1(), EnsembleThreads(), 
              trajectories=10, dt=τ_end/10000.0, maxiters=1e9,
              callback=cb,
              save_everystep=false, 
              save_start=false)

  ts = map(x -> (x.t[end] / normal_j.ω_p), sim.u)
  Is = map(x -> ustrip(u"μA", I_drive(x)), ts)
  println(Is)
  return Is
end

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