SEAQT-based Adsorption Simulation and Machine Learning Toolkit
===============================================================

This repository contains code and data supporting the manuscript:

"Model for Predicting Adsorption Isotherms and the Kinetics of Adsorption via Steepest-Entropy-Ascent Quantum Thermodynamics"  
Adriana Saldana-Robles, Cesar Damian, William T. Reynolds Jr., and Michael R. von Spakovsky (2025)

---------------------------------------------------------------
Overview
---------------------------------------------------------------

This repository includes:

- Monte Carlo simulations for generating adsorption energy eigenstructures using a modified Replica Exchange Wang–Landau (REWL) algorithm.
- Machine Learning routines in Mathematica for extrapolating energy eigenstructures for dilute systems.
- SEAQT-based analysis for non-equilibrium adsorption dynamics and equilibrium isotherms.

---------------------------------------------------------------
Repository Contents
---------------------------------------------------------------

As_GO_on_m.cpp  
> C++ code implementing a modified Replica Exchange Wang–Landau (REWL) algorithm for simulating arsenic adsorption on graphene oxide.  
> Based on original REWL demo code by Thomas Vogel and Ying Wai Li (2013–2019)  
> License: CC BY-SA 4.0 – https://creativecommons.org/licenses/by-sa/4.0/legalcode  
> Attribution is included in the code header.

MC-Sampling_manuscript.nb  
> Mathematica notebook containing:
  - Machine learning routines to extrapolate energy eigenstructures
  - SEAQT implementation to compute non-equilibrium adsorption kinetics
  - Equilibrium isotherms compatible with Langmuir and Freundlich models

/data/  
> Folder with source data from:
  - Monte Carlo simulations (Wang–Landau and Replica Exchange)
  - Training sets for the neural networks used in energy extrapolation
  - SEAQT output datasets for equilibrium and kinetic analysis

---------------------------------------------------------------
How to Use
---------------------------------------------------------------

1. Compile and run `As_GO_on_m.cpp` to generate energy levels and occupation statistics for the adsorption system.
2. Open `MC-Sampling_manuscript.nb` in Wolfram Mathematica to:
   - Train and validate ML models
   - Perform non-equilibrium analysis using SEAQT
   - Generate plots compatible with experimental validation
3. Use the `/data/` folder as input for simulations or to validate results against published datasets.

---------------------------------------------------------------
License
---------------------------------------------------------------

This project is licensed under the Attribution–ShareAlike 4.0 International (CC BY-SA 4.0).  
https://creativecommons.org/licenses/by-sa/4.0/

- You are free to share, adapt, and build upon this material, even for commercial purposes, as long as you provide attribution and license your derivatives under the same terms.
- For modified REWL code, proper attribution to the original authors must be maintained as outlined in the license.

---------------------------------------------------------------
Citing This Work
---------------------------------------------------------------

If you use this repository in your own work, please cite:

Adriana Saldana-Robles, Cesar Damian, William T. Reynolds Jr., and Michael R. von Spakovsky,  
"Model for Predicting Adsorption Isotherms and the Kinetics of Adsorption via Steepest-Entropy-Ascent Quantum Thermodynamics", 2025. (Manuscript in preparation)

And for the original REWL algorithm:

T. Vogel et al., "Generic, hierarchical framework for massively parallel Wang-Landau sampling",  
Phys. Rev. Lett. 110, 210603 (2013)  
T. Vogel et al., Phys. Rev. E 90, 023302 (2014)  
Additional references:
- J. Phys.: Conf. Ser. 487 (2014) 012001
- J. Phys.: Conf. Ser. 510 (2014) 012012

---------------------------------------------------------------
Contact
---------------------------------------------------------------

For questions or collaborations, please contact:

Adriana Saldana-Robles  
Department of Agricultural Engineering, University of Guanajuato  
Email: adriana.saldana@ugto.mx
