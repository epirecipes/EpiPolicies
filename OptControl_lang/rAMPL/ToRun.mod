model lockdown_model.mod
data lockdown_data.dat
option solver "/opt/homebrew/bin/ipopt";
option presolve 0;
option ipopt_options 'output_file=AMPL_IDE_log_presolve_assoc0.txt';
option ipopt_options 'print_timing_statistics=yes';
solve;