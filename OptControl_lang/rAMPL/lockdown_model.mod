# Parameters
param beta;      # Infectivity rate
param gamma;     # Recovery rate
param v_max;     # Maximum intervention
param v_total;   # Maximum cost

param t0;
param tf;
param dt;

param S0;
param I0;
param C0;

param T := round(tf / dt);

# Variables
var S{t in 1..T+1} >= 0, <= 1;   # Susceptible
var I{t in 1..T+1} >= 0, <= 1;   # Infected
var C{t in 1..T+1} >= 0, <= 1;   # Cumulative infections
var v{t in 1..T+1} >= 0, <= v_max;   # Intervention

# Total intervention cost constraint
subject to TotalCostConstraint:
    dt * sum {t in 1..T+1} v[t] <= v_total;

# Nonlinear expressions for infection and recovery rates
var infection{t in 1..T} >= 0;
var recovery{t in 1..T} >= 0;

# Define infection and recovery explicitly as constraints
subject to DefineInfection {t in 1..T}:
    infection[t] = (1 - exp(-(1 - v[t]) * beta * I[t] * dt)) * S[t];

subject to DefineRecovery {t in 1..T}:
    recovery[t] = (1 - exp(-gamma * dt)) * I[t];

# Initial Conditions
subject to InitialConditionS: S[1] == S0;
subject to InitialConditionI: I[1] == I0;
subject to InitialConditionC: C[1] == C0;

# Nonlinear differential equations
subject to SusceptibleConstraint {t in 1..T}:
    S[t+1] == S[t] - infection[t];

subject to InfectedConstraint {t in 1..T}:
    I[t+1] == I[t] + infection[t] - recovery[t];

subject to CumulativeInfectionConstraint {t in 1..T}:
    C[t+1] == C[t] + infection[t];

# Objective: Minimize cumulative infections at the final time
minimize TotalCumulativeInfections:
    C[T+1];

