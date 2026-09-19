"""
structural_identifiability_seir.jl

Structural (algebraic) identifiability analysis of the SEIR model, using
StructuralIdentifiability.jl. This complements the practical,
Fisher-Information-based identifiability diagnosis
(seir_identifiability_analysis.py) with a model-structure-only guarantee:
a structural (non-)identifiability result holds for almost all parameter
values under noise-free, continuous-time, infinite-precision
observations, independent of the specific reference parameter set or
noise realization used elsewhere in this study. Practical (numerical,
FIM-based) identifiability can fail even when a model is structurally
identifiable, if the available data are too sparse, too noisy, or too
short an observation window; the two notions are complementary, and
reporting both is standard practice in the identifiability literature.

Three scenarios are assessed, matching seir_identifiability_analysis.py:
observing I(t) alone; observing I(t) and R(t) jointly; and observing
rho*I(t), with rho itself treated as an unknown to be identified
alongside beta, sigma, and gamma (matching the joint estimation of rho by
NLS and EINN in the real-data experiment, Section 4.5). The population
size N is a known constant (1,000,000, matching the reference synthetic
population used throughout this study), not an unknown to be identified.

To run: install Julia (julialang.org, or via juliaup), install the
package from the REPL with `using Pkg; Pkg.add("StructuralIdentifiability")`,
then `julia structural_identifiability_seir.jl`. For each scenario, the
package reports, for every unknown parameter and state variable, one of
:globally (identifiable from a single experiment), :locally (identifiable
up to a finite number of alternatives), or :nonidentifiable.

Verified: run successfully by the authors on a Julia installation not
available in the main development environment; output reported in the
manuscript's Table 4.
"""

using StructuralIdentifiability

const POPULATION_SIZE = 1_000_000

println("="^70)
println("Scenario 1: observe I(t) only")
println("="^70)

ode_scenario1 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y(t) = I(t)
)

result_scenario1 = assess_identifiability(ode_scenario1)
println(result_scenario1)
println()

println("="^70)
println("Scenario 2: observe I(t) and R(t) jointly")
println("="^70)

ode_scenario2 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y1(t) = I(t),
    y2(t) = R(t)
)

result_scenario2 = assess_identifiability(ode_scenario2)
println(result_scenario2)
println()

println("="^70)
println("Scenario 3: observe rho*I(t), with rho unknown (underreporting,")
println("jointly estimated -- matches the real-data experiment design)")
println("="^70)

ode_scenario3 = @ODEmodel(
    S'(t) = -beta * S(t) * I(t) / 1000000,
    E'(t) = beta * S(t) * I(t) / 1000000 - sigma * E(t),
    I'(t) = sigma * E(t) - gamma * I(t),
    R'(t) = gamma * I(t),
    y(t) = rho * I(t)
)

result_scenario3 = assess_identifiability(ode_scenario3)
println(result_scenario3)
println()

println("="^70)
println("Done. Compare these structural results against the practical")
println("(FIM-based) identifiability diagnosis reported in Table 2 of the")
println("manuscript (condition numbers 4183.8-8427.8, all below the 1e4")
println("practical-identifiability threshold used there) and against the")
println("beta-sigma confounding direction reported in the same table.")
println("="^70)
