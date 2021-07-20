Project Contents:

calc_Mathieu_dos.py:
- using Scipy functions for numerically solving the Mathieu equation, the density of states (DOS) is calculated for integer multiples of the input cyrstal momentum
- import and plot eigenfunction and eigenenergy plots from json files output by functions in the calculate_Mathieu_energies.nb notebook

plot_Moire_2d.py:
- plot the lattice mismatch of a Moire pattern using input lattice constants

calculate_Mathieu_energies.nb:
- Using built in Mathematica functions for calculating eigenenergies and eigenfunctions of the Mathieu equation, 2D plots are computed and written to .json, to be input to Python
- also in the notebook are functions to plot horizontal slices of the eigenenergy (which gives the dispersion relation) and eigenfunctions plots
