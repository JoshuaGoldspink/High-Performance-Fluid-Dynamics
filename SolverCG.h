#pragma once
#include <vector>
#include <mpi.h>
#include <omp.h>

/**
 * @class       SolverCG
 * @brief       SolverCG is a conjugate gradient algorithm to solve the poisson equation
 *              for the streamfunction at time t+delta_t, in a 2D domain. 
 * 
 * 
 * @note        Requires MPI and OpenMP to be run in parallel.
 */
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, int Startx, int Endx, int Starty, int Endy, MPI_Comm comm, int north, int east, int south, int west );
    ~SolverCG();

    void Solve(double* b, double* x);

private:
    double dx;          ///< Domain discretisation in the x-direction (i).
    double dy;          ///< Domain discretisation in the y-direction (j).
    int Nx;             ///< Number of grid points in the x-direction.
    int Ny;             ///< Number of grid points in the y-direction.
    int NxLocal;        ///< Number of grid points in the x-direction of the local sub-domain.
    int NyLocal;        ///< Number of grid points in the y-direction of the local sub-domain.
    double* r;          ///< Residual vector.
    double* p;          ///< Direction vector.
    double* z;          ///< Preconditioned residual vector.
    double* t;          ///< Temporary vector.
    
    MPI_Comm comm;      ///< MPI communicator.
    int world_rank;     ///< Rank of the MPI process within all ranks.
    int world_size;     ///< Total number of MPI processes.
    
    int North;          ///< Integer of the rank of the northern neighbour.
    int South;          ///< Integer of the rank of the southern neighbour
    int East;           ///< Integer of the rank of the eastern neighbour
    int West;           ///< Integer of the rank of the western neighbour
    
    

    void ApplyOperator(double* p, double* t);       // Applies differention
    void Precondition(double* p, double* t);        // Pre-conditions: improves ocnvergences
    void ImposeBC(double* p);                       // Apply problem BCs
    
    // exchange data between ranks at local borders
    void exchangeBoundaryData(double* s, int NxLocal, int NyLocal, int north, int south, int east, int west);
    

};

