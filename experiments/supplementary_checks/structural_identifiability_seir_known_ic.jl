"""
structural_identifiability_seir_known_ic.jl

Refinement of structural_identifiability_seir.jl: repeats the same three
observability scenarios, but with the initial condition declared known
(S(0), E(0), I(0), R(0) all known exactly), matching every fitting
experiment and the FIM diagnostic elsewhere in this study, instead of the
generic, unknown initial condition assumed by default in the original
script. This addresses the caveat noted in the manuscript's Results
(Section 4.1): the original structural analysis is more conservative than
the rest of the study because it does not assume the initial condition is
known.

StructuralIdentifiability.jl supports this via the `known_ic` keyword
argument to `assess_identifiability`, a list of state variables (or
functions of states and parameters) whose initial conditions are assumed
known; see the package documentation
(https://docs.sciml.ai/StructuralIdentifiability/stable/identifiability/identifiability/)
and the epidemiological tutorial at https://arxiv.org/abs/2505.10517, which
demonstrates the same usage for SEIR-type models
(`assess_identifiability(ode, known_ic = [S, E, I, R])`).

To run: `julia structural_identifiability_seir_known_ic.jl`. Compare the
output against structural_identifiability_seir.jl's (Table 4 in the
manuscript) to see how much the identifiability picture improves once the
initial condition is known.

Verified: run successfully by the authors; output reported in the
manuscript's Table 5.
"""

using StructuralIdentifiability

println("="^70)
println("Scenario 1 (known IC): observe I(t) only")
println("="^70)

ode_scenario1 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y(t) = I(t)
)

result_scenario1 = assess_identifiability(ode_scenario1, known_ic = [S, E, I, R])
println(result_scenario1)
println()

println("="^70)
println("Scenario 2 (known IC): observe I(t) and R(t) jointly")
println("="^70)

ode_scenario2 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y1(t) = I(t),
    y2(t) = R(t)
)

result_scenario2 = assess_identifiability(ode_scenario2, known_ic = [S, E, I, R])
println(result_scenario2)
println()

println("="^70)
println("Scenario 3 (known IC): observe rho*I(t), with rho unknown")
println("(the reporting fraction itself is NOT included in known_ic,")
println("since it is precisely the quantity assumed unknown in this")
println("scenario, matching Section 4.5's real-data design)")
println("="^70)

ode_scenario3 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y(t) = rho * I(t)
)

result_scenario3 = assess_identifiability(ode_scenario3, known_ic = [S, E, I, R])
println(result_scenario3)
println()

println("="^70)
println("Done. Compare against the generic-IC results in Table 4 of the")
println("manuscript (structural_identifiability_seir.jl's output) to see")
println("whether R and other states/parameters become identifiable once")
println("the initial condition matches the fixed-IC design used")
println("throughout the rest of this study.")
println("="^70)
