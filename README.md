This is a repository for a coursework on high performance computing in c++. 

The project is a fluid dynamics solver which solves the Lid Driven Cavity problem in 2 dimensions.

It uses a conjugate gradient algorithm to solve the Poisson equation at each timestep. It uses MPI to decompose the domain into sub-domains that are solved concurrently. 
Only border data is exchanged at each timestep. It uses multi-threading with OpenMP for further optimisations. 

A serial version of the code solves the problem with these specifications: 
Lx=1 Ly=1 Nx=201 Ny=201 Re=1000 dt=0.005 T=50

in over an hour. The parallelised version with 16 cores solves it in under 6 minutes.


Further improvements: use BLAS to turn the differentiation function into matrix operations. Currently it uses a finite difference stencil with a nested for loop, but can be optimised with a BLAS matrix multiplication. 


Please let me know if you have any feedback. 

