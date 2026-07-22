# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)

### Set working directory (only works in RStudio)
apollo_setWorkDir()

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelDescr      = "Mixed logit model on Swiss route choice data, uncorrelated Lognormals in preference space",
  indivID         = "ID",  
  nCores          = 4,
  outputDirectory = "output"
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
### if data is to be loaded from a file (e.g. called data.csv), 
### the code would be: database = read.csv("data.csv",header=TRUE)
database = apollo_swissRouteChoiceData
### for data dictionary, use ?apollo_swissRouteChoiceData

# ################################################################# #
#### ANALYSIS OF CHOICES                                         ####
# ################################################################# #

### Note this is unlabelled data, so we are doing this for an illustration of cheap vs expensive
### Define settings for analysis of choice data to be conducted prior to model estimation
choiceAnalysis_settings = list(
  alternatives = c(cheap=1, expensive=2),
  choiceVar    = (1*((database$choice==1)*(database$tc1<=database$tc2)+(database$choice==2)*(database$tc1>=database$tc2))
                 +2*((database$choice==1)*(database$tc1>=database$tc2)+(database$choice==2)*(database$tc1<=database$tc2))),
  explanators  = database[,c("car_availability","hh_inc_abs","business")]
)

### Run function to analyse choice data
apollo_choiceAnalysis(choiceAnalysis_settings, apollo_control, database)

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(mu_log_b_tt    = -3,
                sigma_log_b_tt = 0.01,
                mu_log_b_tc    = -3,
                sigma_log_b_tc = 0.01,
                mu_log_b_hw    = -3,
                sigma_log_b_hw = 0.01,
                mu_log_b_ch    = -3,
                sigma_log_b_ch = 0.01)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c()

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  interDrawsType = "halton",
  interNDraws    = 100,
  interUnifDraws = c(),
  interNormDraws = c("draws_tt","draws_tc","draws_hw","draws_ch"),
  intraDrawsType = "halton",
  intraNDraws    = 0,
  intraUnifDraws = c(),
  intraNormDraws = c()
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()

  randcoeff[["b_tt"]] = -exp( mu_log_b_tt + sigma_log_b_tt * draws_tt )
  randcoeff[["b_tc"]] = -exp( mu_log_b_tc + sigma_log_b_tc * draws_tc )
  randcoeff[["b_hw"]] = -exp( mu_log_b_hw + sigma_log_b_hw * draws_hw )
  randcoeff[["b_ch"]] = -exp( mu_log_b_ch + sigma_log_b_ch * draws_ch )

  return(randcoeff)
}

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){

  ### Function initialisation: do not change the following three commands
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  ### Create list of probabilities P
  P = list()

  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  V[["alt1"]] = b_tt * tt1 + b_tc * tc1 + b_hw * hw1 + b_ch * ch1
  V[["alt2"]] = b_tt * tt2 + b_tc * tc2 + b_hw * hw2 + b_ch * ch2

  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(alt1=1, alt2=2),
    avail         = list(alt1=1, alt2=1),
    choiceVar     = choice,
    utilities     = V
  )

  ### Compute probabilities using MNL model
  P[["model"]] = apollo_mnl(mnl_settings, functionality)

  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)

  ### Average across inter-individual draws
  P = apollo_avgInterDraws(P, apollo_inputs, functionality)

  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed,apollo_probabilities, apollo_inputs)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #

apollo_saveOutput(model)
