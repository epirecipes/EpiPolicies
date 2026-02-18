library(rAMPL)
env <- new(Environment, "/Applications/AMPL/")
ampl <- new(AMPL, env)

# Solver options
ampl$setOption("solver", "ipopt")
ampl$setOption("presolve", "0")
ampl$setOption("presolve_assoc", "0")
ampl$setOption("substout", "0")
ampl$setOption("ipopt_options", "print_timing_statistics=yes output_file=rAMPL_ipopt_log.txt")

# Load model and data
if (!file.exists("lockdown_model.mod")) setwd(file.path(getwd(), "rAMPL"))
ampl$read("lockdown_model.mod")
ampl$readData("lockdown_data.dat")

# Parse IPOPT log for solver metrics
parse_ipopt_metrics <- function(logfile) {
  lines <- readLines(logfile)
  metrics <- list()

  line <- tail(grep("nonzeros in Lagrangian Hessian", lines, value = TRUE), 1)
  metrics$hessian_nnz <- as.numeric(sub(".*:\\s+", "", line))

  line <- tail(grep("nonzeros in equality constraint Jacobian", lines, value = TRUE), 1)
  metrics$eq_jac_nnz <- as.numeric(sub(".*:\\s+", "", line))

  line <- tail(grep("Number of Iterations", lines, value = TRUE), 1)
  metrics$iterations <- as.numeric(sub(".*:\\s+", "", line))

  line <- tail(grep("OverallAlgorithm", lines, value = TRUE), 1)
  metrics$overall_wall <- as.numeric(
    sub(".*wall:\\s+([0-9.]+).*", "\\1", line)
  )

  line <- tail(grep("objective function evaluations", lines, value = TRUE), 1)
  metrics$obj_func_evals <- as.numeric(sub(".*=\\s+", "", line))

  line <- tail(grep("Lagrangian Hessian evaluations", lines, value = TRUE), 1)
  metrics$hessian_evals <- as.numeric(sub(".*=\\s+", "", line))

  line <- tail(grep("^Function Evaluations", lines, value = TRUE), 1)
  metrics$func_eval_wall <- as.numeric(
    sub(".*wall:\\s+([0-9.]+).*", "\\1", line)
  )

  line <- tail(grep("^Objective\\.\\.\\.", lines, value = TRUE), 1)
  metrics$objective <- as.numeric(trimws(sub(".*\\s{2,}", "", line)))

  metrics
}

# Benchmark: 100 samples
log_file <- "rAMPL_ipopt_log.txt"
metric_keys <- c("hessian_nnz", "eq_jac_nnz", "iterations", "overall_wall",
                  "obj_func_evals", "hessian_evals", "func_eval_wall", "objective")
n_samples <- 100
all_metrics <- setNames(lapply(metric_keys, function(x) numeric(n_samples)), metric_keys)

for (i in seq_len(n_samples)) {
  # Reset variable values to prevent warm starting
  ampl$eval("
    let {t in 1..T+1} S[t] := 0;
    let {t in 1..T+1} I[t] := 0;
    let {t in 1..T+1} C[t] := 0;
    let {t in 1..T+1} v[t] := 0;
    let {t in 1..T} infection[t] := 0;
    let {t in 1..T} recovery[t] := 0;
  ")
  ampl$solve()
  m <- parse_ipopt_metrics(log_file)
  for (key in metric_keys) {
    all_metrics[[key]][i] <- m[[key]]
  }
}

# Display metrics
metric_labels <- list(
  c("Lagrangian Hessian nnz", "hessian_nnz"),
  c("Eq. constraint Jacobian nnz", "eq_jac_nnz"),
  c("Iterations", "iterations"),
  c("Overall Algorithm (wall, s)", "overall_wall"),
  c("Objective function evals", "obj_func_evals"),
  c("Lagrangian Hessian evals", "hessian_evals"),
  c("Function eval time (wall, s)", "func_eval_wall"),
  c("Objective value", "objective")
)

cat(sprintf("rAMPL + Ipopt  (n=%d runs)\n", n_samples))
cat(paste(rep("\u2500", 55), collapse = ""), "\n")
for (item in metric_labels) {
  label <- item[1]
  key <- item[2]
  vals <- all_metrics[[key]]
  mu <- mean(vals)
  sigma <- sd(vals)
  if (is.na(sigma) || sigma / (abs(mu) + 1e-15) < 1e-10) {
    if (!is.na(mu) && mu == round(mu)) {
      cat(sprintf("  %-34s %d\n", label, as.integer(mu)))
    } else if (!is.na(mu)) {
      cat(sprintf("  %-34s %s\n", label, format(mu, digits = 16)))
    } else {
      cat(sprintf("  %-34s NA\n", label))
    }
  } else {
    cat(sprintf("  %-34s %.4f \u00b1 %.4f\n", label, mu, sigma))
  }
}

# Retrieve results
T_val <- ampl$getParameter("T")$value()
dt_val <- ampl$getParameter("dt")$value()
v_total_val <- ampl$getParameter("v_total")$value()
v_max_val <- ampl$getParameter("v_max")$value()
ts <- seq(0, 100, by = dt_val)

S_opt <- sapply(1:(T_val + 1), function(i) ampl$getVariable("S")$get(i)$value())
I_opt <- sapply(1:(T_val + 1), function(i) ampl$getVariable("I")$get(i)$value())
C_opt <- sapply(1:(T_val + 1), function(i) ampl$getVariable("C")$get(i)$value())
v_opt <- sapply(1:(T_val + 1), function(i) ampl$getVariable("v")$get(i)$value())

# Calculate exact control time bounds
t1 <- 14.338623046875002
t2 <- t1 + v_total_val / v_max_val

# Plotting
plot(ts, S_opt, type = "l", col = "blue", ylim = c(0, 1),
     xlab = "Time", ylab = "Population / Control",
     main = "Optimised SIR Model with Control")
lines(ts, I_opt, col = "orange")
lines(ts, C_opt, col = "green")
lines(ts, v_opt, col = "purple", lty = 2)
rect(t1, 0, t2, 1, col = rgb(0.5, 0.5, 0.5, 0.3), border = NA)
legend("right", legend = c("S", "I", "C", "Optimised v", "Exact v"),
       col = c("blue", "orange", "green", "purple", "gray"),
       lty = c(1, 1, 1, 2, 1), lwd = c(1, 1, 1, 1, 8))
grid()

# Save to CSV
data <- data.frame(
  timestep = 1:(T_val + 1),
  S = S_opt,
  I = I_opt,
  C = C_opt,
  v = v_opt
)
write.csv(data, "rAMPL_results.csv", row.names = FALSE)

ampl$close()

# Overall results:
# Ipopt 3.14.19: Optimal Solution Found
# rAMPL + Ipopt  (n=100 runs)
# ─────────────────────────────────────────────────────── 
#   Lagrangian Hessian nnz             1000
#   Eq. constraint Jacobian nnz        3203
#   Iterations                         128
#   Overall Algorithm (wall, s)        0.1598 ± 0.0078
#   Objective function evals           183
#   Lagrangian Hessian evals           128
#   Function eval time (wall, s)       0.0092 ± 0.0006
#   Objective value                    0.6332064764158756