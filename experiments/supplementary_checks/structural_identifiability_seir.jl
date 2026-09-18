"""
structural_identifiability_seir.jl

Structural (algebraic) identifiability analysis of the SEIR model, using
StructuralIdentifiability.jl. This complements the practical, Fisher-
Information-based identifiability diagnosis (seir_identifiability_analysis.py)
with a model-structure-only guarantee: a structural (non-)identifiability
result holds for ALMOST ALL parameter values and noise-free, continuous-time,
infinite-precision observations, independent of the specific reference
parameter set or noise realization used elsewhere in this study. Practical
(numerical, FIM-based) identifiability, in contrast, can fail even when a
model is structurally identifiable, if the available data are too sparse,
too noisy, or too short an observation window -- the two notions are
complementary, not redundant, and reporting both is standard practice in
the identifiability literature.

THREE SCENARIOS, MATCHING seir_identifiability_analysis.py
------------------------------------------------------------
1. I only:          y(t) = I(t)
2. I and R:          y1(t) = I(t), y2(t) = R(t)
3. I underreported:  y(t) = rho * I(t), with rho itself treated as an
                      additional unknown to be assessed for identifiability
                      jointly with beta, sigma, gamma (directly relevant to
                      the real-data experiment, Section 4.5, where rho is
                      estimated jointly for NLS and EINN).

The population size N is treated as a known constant (1,000,000, matching
the reference synthetic population used throughout this study), not as an
unknown to be identified, consistent with every other experiment in this
study.

HOW TO RUN
-----------
1. Install Julia (https://julialang.org/downloads/, or via juliaup).
2. From the Julia REPL, install the package (one-time):
       using Pkg
       Pkg.add("StructuralIdentifiability")
3. Run this script:
       julia structural_identifiability_seir.jl

EXPECTED OUTPUT
-----------------
For each scenario, StructuralIdentifiability.jl reports, for every unknown
parameter (and, where relevant, every state variable), one of:
    :globally    -- identifiable from a single experiment, uniquely
    :locally     -- identifiable up to a finite number of alternatives
    :nonidentifiable -- not identifiable from noise-free, continuous data,
                         regardless of how much data is collected

IMPORTANT NOTE ON VERIFICATION
---------------------------------
This script was written without access to a Julia installation in the
development environment used for the rest of this study (see the
manuscript's Limitations section), and has NOT been executed or verified
end-to-end prior to being shared. Please run it and report any errors;
the model equations themselves have been checked by hand against Eq. (1)-(4)
of the manuscript, but the StructuralIdentifiability.jl syntax has not been
confirmed against a live installation.

LICENSE
--------
Intended to be released alongside the associated publication under an open
license (e.g., MIT), consistent with the rest of the code in this study.
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
