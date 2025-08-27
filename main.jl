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
  sim = solve(ensemble_prob,SRA1(), EnsembleThreads(), 
              trajectories=10, dt=0.1, maxiters=1e9,
              callback=cb,
              save_everystep=false, 
              save_start=false)

  ts = map(x -> (x.t[end] / normal_j.ω_p), sim.u)
  Is = map(x -> ustrip(u"μA", I_drive(x)), ts)

  return Is
end

Iss = map(run_with_T, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] .* 100.0u"mK")
println(map(mean, Iss))
println(map(std, Iss))

return Iss

end