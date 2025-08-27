module JosephsonJunction
  using DifferentialEquations
  using PhysicalConstants.CODATA2022
  using Unitful

  export run_chaos_sim, gen_nois_prob, normalize_josephson, get_noise_gamma

  const ħ = ReducedPlanckConstant
  const e = ElementaryCharge
  const k_B = BoltzmannConstant

  function rcsj_equation!(du, u, p, τ)
    ϕ, ω = u
    β, drive_func = p

    i_drive = drive_func(τ)

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
    sol = solve(prob, TsitPap8(), abstol = 1e-8, reltol = 1e-8)

    return sol
  end

  function gen_nois_prob(p, tau_end)
    prob = SDEProblem(rcsj_equation!, rcsj_noise!, [0,0], [0.0, tau_end], p)
    
    condition(u, t, integrator) = u[1] - pi / 2
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    return (prob, cb)
  end

  function normalize_josephson(p)
    β = 1 / sqrt(2 * e / ħ * p.I_c * p.R * p.R * p.C)
    ω_p = sqrt(2 * e * p.I_c / (ħ * p.C))
    i_drive(τ) = upreferred(p.I_drive(τ / ω_p) / p.I_c)

    (β = upreferred(β), ω_p = upreferred(ω_p), i_drive = i_drive)
  end

  function get_noise_gamma(p)
    upreferred((2 * e * k_B * p.T) / (ħ * p.I_c))
  end

end

