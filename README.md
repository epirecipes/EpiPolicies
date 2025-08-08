# EpiPolicies
Optimal control for infectious disease epidemics.

## Introduction

This repository contains implementations of optimal control problems for infectious disease epidemics, showcasing multiple epidemiological models and optimisation frameworks. It provides a comparison of different approaches to solving epidemiological optimisation problems.

**Related preprint is now available:**  
“Exploring epidemic control policies using nonlinear programming and mathematical models”
[Read it on arXiv](https://arxiv.org/abs/2508.05290)

## Examples

The following examples demonstrate the application of [JuMP](https://github.com/jump-dev/JuMP.jl) ([Julia](https://julialang.org/)), in combination with the [IPOPT](https://github.com/coin-or/Ipopt) solver, to assess control intervention strategies within a Susceptible-Infected-Recovered (SIR) model. These examples assess various strategic objectives, such as minimising total infections during an epidemic through lockdown measures, "flattening the curve", or vaccination efforts.

- [SIR model with optimal lockdown timing and intensity](https://github.com/epirecipes/EpiPolicies/blob/main/NPIs/Lockdown/SIR_lockdown_JuMP.md) This example is used to demostrate implementations across multiple optimisation frameworks:
    - [Pyomo (Python)](https://github.com/epirecipes/EpiPolicies/blob/main/OptControl_lang/Python/Python_pyomo_lockdown.ipynb)
    - [AmplNLWriter (Julia interface to AMPL)](https://github.com/epirecipes/EpiPolicies/blob/main/OptControl_lang/Julia/Julia_AmplNLWriter_lockdown.ipynb)
    - [rAMPL (R interface to AMPL)](https://github.com/epirecipes/EpiPolicies/blob/main/OptControl_lang/rAMPL/lockdown_rAMPL.R)

- [SIR model with an optimal policy strategy to keep the number of infected individuals below a threshold, commonly referred to as "flattening the curve''](https://github.com/epirecipes/EpiPolicies/blob/main/NPIs/FlatteningTheCurve/SIR_ftc_JuMP.md)

- [SIR model with optimal vaccine allocation and timing](https://github.com/epirecipes/EpiPolicies/blob/main/Vaccination/SIR_vaccination_JuMP.md)

- [Compartmental model to evaluate the optimal combination of multiple control strategies to control dengue transmission](https://github.com/epirecipes/EpiPolicies/blob/main/MultipleControl/MultControl_Dengue.md)

## Contributing

We welcome contributions to this repository. Each example should be placed in its own directory, as a Quarto (`.qmd`) file. The header of the notebook should contain a list of output formats, for example:

```yaml
---
title: Lockdown optimisation on an SIR model using JuMP.jl
date: 2025-03-10
author: Simon Frost (@sdwfrost) and Sandra Montes (@slmontes)
format:
    html: default
    docx: default
    gfm: default
    pdf: default
---
```

To render Quarto files, install Quarto and run the following command in the example directory (where `{FILENAME_OF_QMD}` is the name of the Quarto file):

```bash
quarto render {FILENAME_OF_QMD}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.