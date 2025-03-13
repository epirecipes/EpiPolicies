library(rAMPL)
env <- new(Environment, "/Applications/AMPL/")
ampl <- new(AMPL, env)

# Provide the full path to the test model
ampl$setOption("solver", "/opt/homebrew/bin/ipopt") 
ampl$setOption("presolve", "0")
ampl$setOption("log_file", "ampl_fromR_presolve0.log")
ampl$setOption("ipopt_options", "print_timing_statistics=yes");
ampl$read("lockdown_model.mod")
ampl$readData("lockdown_data.dat")
ampl$read("lockdown_runR.mod")
