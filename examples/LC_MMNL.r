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
  modelDescr      = "Latent class with continuous random parameters on Swiss route choice data",
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
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta = c(asc1            =  0,
                asc2            =  0,
                log_tt_a_mu     =  -3,
                log_tt_b_mu     =  -3,
                log_tt_a_sig    =  0.1,
                log_tt_b_sig    =  0.1,
                tc_a            =  -0.1,
                tc_b            =  -0.2,
                hw_a            = -0.1,
                hw_b            = -0.2,
                ch_a            = -1,
                ch_b            = -2,
                delta_a         =  0,
                gamma_commute_a =  0,
                gamma_car_av_a  =  0,
                delta_b         =  0,
                gamma_commute_b =  0,
                gamma_car_av_b  =  0)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc2", "delta_b", "gamma_commute_b","gamma_car_av_b")

# ################################################################# #
#### DEFINE RANDOM COMPONENTS                                    ####
# ################################################################# #

### Set parameters for generating draws
apollo_draws = list(
  interDrawsType="halton",           
  interNDraws=500,                   
  interUnifDraws=c(),                
  interNormDraws=c("draws_tt"),    
  
  intraDrawsType="mlhs",
  intraNDraws=0,
  intraUnifDraws=c(),
  intraNormDraws=c()
)

### Create random parameters
apollo_randCoeff = function(apollo_beta, apollo_inputs){
  randcoeff = list()
  
  randcoeff[["tt_a"]] = -exp(log_tt_a_mu + log_tt_a_sig*draws_tt)
  randcoeff[["tt_b"]] = -exp(log_tt_b_mu + log_tt_b_sig*draws_tt)

  return(randcoeff)
}

# ################################################################# #
#### DEFINE LATENT CLASS COMPONENTS                              ####
# ################################################################# #

apollo_lcPars = function(apollo_beta, apollo_inputs){
  lcpars = list()
  lcpars[["tt"]] = list(tt_a, tt_b)
  lcpars[["tc"]] = list(tc_a, tc_b)
  lcpars[["hw"]] = list(hw_a, hw_b)
  lcpars[["ch"]] = list(ch_a, ch_b)
  
  V=list()
  V[["class_a"]] = delta_a + gamma_commute_a*commute + gamma_car_av_a*car_availability
  V[["class_b"]] = delta_b + gamma_commute_b*commute + gamma_car_av_b*car_availability
  
  classAlloc_settings = list(
    classes      = c(class_a=1, class_b=2), 
    utilities    = V
  )
  
  lcpars[["pi_values"]] = apollo_classAlloc(classAlloc_settings)
  
  return(lcpars)
}

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
    
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  ### Create list of probabilities P
  P = list()
  
  ### Define settings for MNL model component that are generic across classes
  mnl_settings = list(
    alternatives = c(alt1=1, alt2=2),
    avail        = list(alt1=1, alt2=1),
    choiceVar    = choice
  )
  
  ### Loop over classes
  for(s in 1:2){
    ### Compute class-specific utilities
    V=list()
    V[["alt1"]]  = asc1 + tc[[s]]*tc1 + tt[[s]]*tt1 + hw[[s]]*hw1 + ch[[s]]*ch1
    V[["alt2"]]  = asc2 + tc[[s]]*tc2 + tt[[s]]*tt2 + hw[[s]]*hw2 + ch[[s]]*ch2
    
    mnl_settings$utilities = V
    
    ### Compute within-class choice probabilities using MNL model
    P[[paste0("Class_",s)]] = apollo_mnl(mnl_settings, functionality)
    
    ### Take product across observation for same individual
    P[[paste0("Class_",s)]] = apollo_panelProd(P[[paste0("Class_",s)]], apollo_inputs ,functionality)

    ### Average across inter-individual draws within classes
    P[[paste0("Class_",s)]] = apollo_avgInterDraws(P[[paste0("Class_",s)]], apollo_inputs, functionality)
  }
  
  ### Compute latent class model probabilities
  lc_settings  = list(inClassProb = P, classProb=pi_values)
  P[["model"]] = apollo_lc(lc_settings, apollo_inputs, functionality)

  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}


# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

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
