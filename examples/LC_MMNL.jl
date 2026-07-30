using CSV, DataFrames, Statistics, Random
using ChoiceModels

# Latent class with CONTINUOUS random parameters inside each class (LC-MMNL), on
# the Swiss route choice data. Two classes, each a Mixed Logit with a lognormal
# travel-time coefficient, plus a class-allocation model with socio-demographics.
#
# This is the Julia port of `LC_MMNL.r` (Apollo), which is the reference for the
# numbers below — see the comparison note at the bottom of the file.
#
# The likelihood is
#
#     L_i = Σ_c π_c · (1/R) Σ_r Π_t P_c(j_t | β_cr)
#
# i.e. the product over an individual's observations is innermost, then the draws
# are averaged, then the classes are mixed. Both the class membership and the
# random coefficients belong to the INDIVIDUAL, which is why `idvar` is required.

df = CSV.read("../data/apollo_swissRouteChoiceData.csv", DataFrame)
df = sort(df, :ID)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
asc_1 = Parameter(:asc1, value=0.0)
asc_2 = Parameter(:asc2, value=0.0, fixed=true)

# Lognormal travel-time coefficients, one pair of (mu, sigma) per class.
log_tt_a_mu  = Parameter(:log_tt_a_mu,  value=-3.0)
log_tt_b_mu  = Parameter(:log_tt_b_mu,  value=-3.0)
log_tt_a_sig = Parameter(:log_tt_a_sig, value=0.1)
log_tt_b_sig = Parameter(:log_tt_b_sig, value=0.1)

tc_a = Parameter(:tc_a, value=-0.1);  tc_b = Parameter(:tc_b, value=-0.2)
hw_a = Parameter(:hw_a, value=-0.1);  hw_b = Parameter(:hw_b, value=-0.2)
ch_a = Parameter(:ch_a, value=-1.0);  ch_b = Parameter(:ch_b, value=-2.0)

# Class allocation: class b is the reference, so all of its parameters are fixed.
delta_a         = Parameter(:delta_a,         value=0.0)
gamma_commute_a = Parameter(:gamma_commute_a, value=0.0)
gamma_car_av_a  = Parameter(:gamma_car_av_a,  value=0.0)
delta_b         = Parameter(:delta_b,         value=0.0, fixed=true)
gamma_commute_b = Parameter(:gamma_commute_b, value=0.0, fixed=true)
gamma_car_av_b  = Parameter(:gamma_car_av_b,  value=0.0, fixed=true)

# ONE draw dimension, referenced by BOTH classes. Sharing is by symbol: the same
# `Draw` name in two classes is the same draw, and the latent class model
# generates it once for all of them. This mirrors Apollo, where a single
# `interNormDraws = c("draws_tt")` feeds both classes' random coefficients.
draws_tt = Draw(:draws_tt)

b_tt_a = -exp(log_tt_a_mu + log_tt_a_sig * draws_tt)
b_tt_b = -exp(log_tt_b_mu + log_tt_b_sig * draws_tt)

# ---------------------------------------------------------------------------
# The two classes
# ---------------------------------------------------------------------------
alternatives = (route1 = 1, route2 = 2)
availability = (route1 = trues(nrow(df)), route2 = trues(nrow(df)))

utilities_a = (
    route1 = asc_1 + tc_a * Variable(:tc1) + b_tt_a * Variable(:tt1) +
             hw_a * Variable(:hw1) + ch_a * Variable(:ch1),
    route2 = asc_2 + tc_a * Variable(:tc2) + b_tt_a * Variable(:tt2) +
             hw_a * Variable(:hw2) + ch_a * Variable(:ch2),
)

utilities_b = (
    route1 = asc_1 + tc_b * Variable(:tc1) + b_tt_b * Variable(:tt1) +
             hw_b * Variable(:hw1) + ch_b * Variable(:ch1),
    route2 = asc_2 + tc_b * Variable(:tc2) + b_tt_b * Variable(:tt2) +
             hw_b * Variable(:hw2) + ch_b * Variable(:ch2),
)

Random.seed!(12345)
class_a = MixedLogitModel(alternatives; utilities=utilities_a, availability=availability,
                          data=df, idvar=:ID, R=500, draw_scheme=:halton)
class_b = MixedLogitModel(alternatives; utilities=utilities_b, availability=availability,
                          data=df, idvar=:ID, R=500, draw_scheme=:halton)

# ---------------------------------------------------------------------------
# Class allocation
# ---------------------------------------------------------------------------
V_class_a = delta_a + gamma_commute_a * Variable(:commute) + gamma_car_av_a * Variable(:car_availability)
V_class_b = delta_b + gamma_commute_b * Variable(:commute) + gamma_car_av_b * Variable(:car_availability)

pi_a = exp(V_class_a) / (exp(V_class_a) + exp(V_class_b))
pi_b = exp(V_class_b) / (exp(V_class_a) + exp(V_class_b))

lc_model = LatentClassModel(pi_a * class_a + pi_b * class_b; data=df, idvar=:ID)
results = estimate(lc_model, :choice)

summarize_results(results, file="output/LC_MMNL.xlsx")

# Apollo reference (`LC_MMNL.r`, apollo 0.3.9, 500 halton draws):
#   LL(final) = -1497.87,  LL(0) = -2420.47,  14 estimated parameters
#   asc1 -0.0302 | log_tt_a_mu -3.1344, log_tt_a_sig 0.7460
#                | log_tt_b_mu -1.5992, log_tt_b_sig 0.5719
#   tc_a -0.0839, tc_b -0.8183 | hw_a -0.0427, hw_b -0.0560
#   ch_a -0.7031, ch_b -2.9451
#   delta_a -0.3400, gamma_commute_a -0.1102, gamma_car_av_a 0.4918
#
# Measured agreement: every estimate, every classical standard error AND every
# robust standard error match Apollo to within 6.1e-5 — at or below the printed
# precision — and the log-likelihood matches to all four decimals. That is much
# closer than a simulated likelihood strictly guarantees; it holds because both
# packages build per-individual blocks of a base-2 Halton sequence for the single
# random dimension, so the two simulated likelihoods are the same function.
#
# The ROBUST standard errors matching is worth noting. They do NOT match for the
# MNL and NL examples, because Apollo's `apollo_panelProd` gives one likelihood
# contribution per individual there while ours are per observation (see item 3 of
# the TODO section in CLAUDE.md — a deliberate difference, not a gap). A panel
# latent class returns one contribution per individual on both sides, so the two
# agree here. The asymmetry is the models' structure, exactly as documented.
#
# Class labels can still swap between runs and between packages; that is inherent
# to latent class models and shows up as class a and b trading estimates.
