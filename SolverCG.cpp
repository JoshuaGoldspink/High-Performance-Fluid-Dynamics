#include <iostream>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cblas.h>
#include "SolverCG.h"

using namespace std;



/*!*****************************************************************************
 * @brief  Define the indexing operations for both local and global matrices, in row-major format
 * @note  indexing from (0,0) in bottom left corner.
 *******************************************************************************/
#define IDX(I,J) ((J)*Nx + (I))
#define IDX_LOCAL(I,J) ((J)*NxLocal + (I))

/**
 * @brief Constructs the SolverCG class.
 * 
 * Initialises local variables, calculates local domain size. 
 *
 * @param pNx           Global number of grid points along x.
 * @param pNy           Global number of grid points along y.
 * @param pdx           Grid spacing along x.
 * @param pdy           Grid spacing along y.
 * @param Startx        Starting index in x for the local domain.
 * @param Endx          Ending index in x for the local domain.
 * @param Starty        Starting index in y for the local domain.
 * @param Endy          Ending index in y for the local domain.
 * @param pcomm         MPI communicator.
 * @param north         Rank of the northern neighbor (-2 if none).
 * @param east          Rank of the eastern neighbor (-2 if none).
 * @param south         Rank of the southern neighbor (-2 if none).
 * @param west          Rank of the western neighbor (-2 if none).
 */
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, int Startx, int Endx, int Starty, int Endy, MPI_Comm pcomm, 
                    int north, int east, int south, int west)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    NxLocal = Endx - Startx + 1;
    NyLocal = Endy - Starty + 1;
    
    int n_local = NxLocal * NyLocal;
    
    r = new double[n_local];
    p = new double[n_local];
    z = new double[n_local];
    t = new double[n_local]; //temp
    
    comm = pcomm;
    MPI_Comm_rank(pcomm, &world_rank);
    MPI_Comm_size(pcomm, &world_size);
    
    North = north;
    East = east;
    South = south;
    West = west;
    
}

/**
 *  @brief Destroys the SolverCG object and deallocates dynamic arrays.
 */
SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
}

/**
 * @brief Exchanges data between ranks.
 *
 * Key function yto exchange the LOCAL borders data across each rank. 
 * Each rank only sends data to the neighbours that exist (!= -2).
 * They send the entire row/column, including corners, because in the case where a rank
 * is at a global border, it needs that corner value (which is also 
 * at a global boundary) for accuracy.
 * 
 * 
 * @param   s           pointer to matrix we wish to print.
 * @param   NxLocal     Size of matrix in the x direction
 * @param   NyLocal     Size of matrix in the y direction
 * @param   north       rank of north neighbour
 * @param   south       rank of south neighbour
 * @param   east        rank of east neighbour
 * @param   west        rank of west neighbour
 */
void SolverCG::exchangeBoundaryData(double* s, int NxLocal, int NyLocal, int north, int south, int east, int west) {
    MPI_Status status;

    // Edge buffers for send/receive operations, accounting for corners when on domain edges
    double* sendNorth = new double[NxLocal];
    double* recvNorth = new double[NxLocal];
    double* sendSouth = new double[NxLocal];
    double* recvSouth = new double[NxLocal];
    double* sendEast = new double[NyLocal]; 
    double* recvEast = new double[NyLocal];
    double* sendWest = new double[NyLocal];
    double* recvWest = new double[NyLocal];

    // Fill send buffers, including handling corners specifically for ranks on edges
    for (int i = 0; i < NxLocal; ++i) {
        sendSouth[i] = s[IDX_LOCAL(i, 1)];              // Send the first real row for south, including corners
        sendNorth[i] = s[IDX_LOCAL(i, NyLocal - 2)];    // Send the last real row for north, including corners
    }
    for (int j = 0; j < NyLocal; ++j) {
        sendWest[j] = s[IDX_LOCAL(1, j)];               // Send the first real column for west, including corners
        sendEast[j] = s[IDX_LOCAL(NxLocal - 2, j)];     // Send the last real column for east, including corners
    }

    // North-South communication
    if (north != -2) {
        MPI_Sendrecv(sendNorth, NxLocal, MPI_DOUBLE, north, 0,
                     recvNorth, NxLocal, MPI_DOUBLE, north, 0, MPI_COMM_WORLD, &status);
                     
        // Place received data into north ghost row
        for (int i = 0; i < NxLocal; ++i) {
            s[IDX_LOCAL(i, NyLocal - 1)] = recvNorth[i];
        }
    }
    if (south != -2) {
        MPI_Sendrecv(sendSouth, NxLocal, MPI_DOUBLE, south, 0,
                     recvSouth, NxLocal, MPI_DOUBLE, south, 0, MPI_COMM_WORLD, &status);
                     
        // Place received data into south ghost row
        for (int i = 0; i < NxLocal; ++i) {
            s[IDX_LOCAL(i, 0)] = recvSouth[i];
        }
    }

    // East-West communication
    if (east != -2) {
        MPI_Sendrecv(sendEast, NyLocal, MPI_DOUBLE, east, 0,
                     recvEast, NyLocal, MPI_DOUBLE, east, 0, MPI_COMM_WORLD, &status);
                     
        // Place received data into east ghost column
        for (int j = 0; j < NyLocal; ++j) {
            s[IDX_LOCAL(NxLocal - 1, j)] = recvEast[j];
        }
    }
    if (west != -2) {
        MPI_Sendrecv(sendWest, NyLocal, MPI_DOUBLE, west, 0,
                     recvWest, NyLocal, MPI_DOUBLE, west, 0, MPI_COMM_WORLD, &status);

        // Place received data into west ghost column
        for (int j = 0; j < NyLocal; ++j) {
            s[IDX_LOCAL(0, j)] = recvWest[j];
        }
    }

    // Cleanup
    delete[] sendNorth;
    delete[] recvNorth;
    delete[] sendSouth;
    delete[] recvSouth;
    delete[] sendEast;
    delete[] recvEast;
    delete[] sendWest;
    delete[] recvWest;
}

/**
 * @brief Prints a matrix.
 *
 * Mainly used for debugging and evaluating the final result for verification.
 * 
 * Assumes row-major indexing, with 0,0 in the bottom left corner.
 * 
 * @param   array   pointer to matrix we wish to print.
 * @param   Nx      Size of matrix in the x direction
 * @param   Ny      Size of matrix in the y direction
 */
void printMatrix2(const double* array, int Nx, int Ny) {
    for (int j = Ny - 1; j >= 0; --j) {
        for (int i = 0; i < Nx; ++i) {
            int index = j * Nx + i;
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << array[index] << " ";
        }
        std::cout << std::endl; // Newline after each row
    }
}

/**
 * @brief Zeros out ghost cells that are not at global boundaries.
 * 
 * Global boundary is known if it is not a local boundary.
 * 
 * @param data          pointer to Local matrix.
 * @param NxLocal       Number of local grid points in x including ghosts.
 * @param NyLocal       Number of local grid points in y including ghosts.
 * @param north         Rank of the northern neighbor (-2 if none).
 * @param south         Rank of the southern neighbor (-2 if none).
 * @param east          Rank of the eastern neighbor (-2 if none).
 * @param west          Rank of the western neighbor (-2 if none).
 */
void zeroGhostCells(double* data, int NxLocal, int NyLocal, int north, int south, int east, int west) {
    
    //Loop over both boundary columns and rows if the rank has a global boundary there
    if (west != -2) { 
        for (int j = 0; j < NyLocal; ++j) {
            data[IDX_LOCAL(0, j)] = 0.0; 
        }
    }
    if (east != -2) { 
        for (int j = 0; j < NyLocal; ++j) {
            data[IDX_LOCAL(NxLocal-1, j)] = 0.0;
        }
    }

    // Loop over the first and last rows
    if (south != -2) { 
        for (int i = 0; i < NxLocal; ++i) {
            data[IDX_LOCAL(i, 0)] = 0.0; 
        }
    }
    if (north != -2) {
        for (int i = 0; i < NxLocal; ++i) {
            data[IDX_LOCAL(i, NyLocal-1 )] = 0.0;
        }
    }
}

/**
 * @brief Solves the system using the Conjugate Gradient method.
 * 
 * This algorithm solves the partial differential equation relating the streamfunction and the vorticity of a flow. 
 * The PDE itself is a second spatial derivative of vorticity.
 * 
 * @param b         local right-hand side vector: the vorticity after time-evolving
 * @param x         local solution vector: the stream function
 */
void SolverCG::Solve(double* b, double* x) {
    
    // Initialise variables
    unsigned int n = NxLocal*NyLocal;
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;
    
    // Clean the b matrix so that the un-updated borders don't affec calculations
    zeroGhostCells(b, NxLocal, NyLocal, North, South, East, West);
    
    // Calculate error: local then global
    eps = cblas_dnrm2(n, b, 1);
    eps = eps * eps;
    
    // Reduce for global error
    MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    eps = sqrt(eps);
   
    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        if (world_rank == 0) cout << "Global norm is below tolerance: " << eps << endl;
        return;
    }
    
    // Update padding for x
    exchangeBoundaryData(x, NxLocal, NyLocal, North, South, East, West);
    
    // Apply Differentiator operation
    ApplyOperator(x, t);
    
    // Initialise r0
    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    
    // Impose Boundary counditions
    ImposeBC(r);
    
    // Calculate t
    cblas_daxpy(n, -1.0, t, 1, r, 1);
        
    // Pre-condition for convergence and update padding
    Precondition(r, z);
    exchangeBoundaryData(z, NxLocal, NyLocal, North, South, East, West);

    // Initialise p0
    cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0

    k = 0;
    // Main loop for converging to a solution
    do {
        k++;
        
        // Perform action of Nabla^2 * p
        ApplyOperator(p, t);

        //Must set ghost cells to zero to not modify the accuracy of the dot product operation
        zeroGhostCells(p, NxLocal, NyLocal, North, South, East, West);
        
        // Variable for the global t dot p
        double global_dot = cblas_ddot(n, t, 1, p, 1);
        MPI_Allreduce(&global_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sum up for total global dot
        
        // Calculate local then global alphas and betas
        alpha = cblas_ddot(n, r, 1, z, 1); //local contribution to alpha
        MPI_Allreduce(&alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sum alphas
        alpha = alpha / global_dot;
        
        
        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
        MPI_Allreduce(&beta, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sum betas
        
        
        // Update x and r to the next iteration
        cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k
        
        // Calculate new error
        eps = cblas_dnrm2(n, r, 1);
        eps = eps * eps;
        MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // Sum local errors
        eps = sqrt(eps);

        // Break clause: converged if the error is small enough
        if (eps < tol*tol) {
            break;
        }
        
        // Pre-condition the residual vector
        Precondition(r, z);
        
        
        // Compute global beta
        double beta_temp = cblas_ddot(n, r, 1, z, 1);
        MPI_Allreduce(&beta_temp, &beta_temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sum betas
        beta = beta_temp / beta;
        
        // Use blas for copying and updating variables before next iteration
        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);
        
        
        // Update all the paddings so everything is accurate and up-to-date for nextr iteration
        exchangeBoundaryData(p, NxLocal, NyLocal, North, South, East, West);
        
       

    } while (k < 5000); // Set a maximum number of iterations

    // Break condition if it fails to converge
    if (k == 5000) {
        if (world_rank == 0)
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }
    
    // Output convergence info to terminal
    if (world_rank == 0) cout << "Converged in " << k << " iterations. eps = " << eps << endl;

}

/**
 * @brief Applies the operator nabla^2 (second spatial derivative).
 * 
 * 
 * @param in        Input vector (dynamically allocated of size NxLocal * NyLocal)
 * @param out       Output vector after differentiation, of same size as in.
 */
void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    
    // Second spatial derivative formula
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < NyLocal - 1; ++j) {
        for (int i = 1; i < NxLocal - 1; ++i) {
            out[IDX_LOCAL(i,j)] = ( -     in[IDX_LOCAL(i-1, j)]
                                  + 2.0*in[IDX_LOCAL(i,   j)]
                                  -     in[IDX_LOCAL(i+1, j)])*dx2i
                              + ( -     in[IDX_LOCAL(i, j-1)]
                                  + 2.0*in[IDX_LOCAL(i,   j)]
                                  -     in[IDX_LOCAL(i, j+1)])*dy2i;
        }
    }
}

/**
 * @rief  Pre-conditions the matrix for better convergence.
 * 
 * Applies a simple jacobian pre-conditioner. The resulting system has the exact same solution,
 * but the convergence is easier with this new system. Simply divide every value of
 * the domain by the same factor, 2* (1/dx^2 + 1/dy^2).
 * 
 * @param in        Input vector.
 * @param out       Output vector after pre-conditioning.
 */
void SolverCG::Precondition(double* in, double* out) {
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    
    // Modify factor so that it works with cblas_dscal, and because multiplication is faster than division
    double factor = 1.0 / (2.0*(dx2i + dy2i));      
    
    // Instead of using for-loops, use blas which is much more efficient
    cblas_dcopy(NxLocal*NyLocal, in, 1, out, 1);
    cblas_dscal(NxLocal*NyLocal,  factor, out, 1);
    

    // Account for cases where rank is at a global border so we dont violate BCs
    if (South == -2) { 
        for (int i = 0; i < NxLocal; ++i) {
            out[IDX_LOCAL(i, 0)] = in[IDX_LOCAL(i, 0)];
        }
    }
    if (North == -2) {
        for (int i = 0; i < NxLocal; ++i) {
            out[IDX_LOCAL(i, NyLocal - 1)] = in[IDX_LOCAL(i, NyLocal - 1)];
        }
    }
    if (West == -2) { 
        for (int j = 0; j < NyLocal; ++j) {
            out[IDX_LOCAL(0, j)] = in[IDX_LOCAL(0, j)];
        }
    }
    if (East == -2) {
        for (int j = 0; j < NyLocal; ++j) {
            out[IDX_LOCAL(NxLocal - 1, j)] = in[IDX_LOCAL(NxLocal - 1, j)];
        }
    }
}

/**
 * @brief Applies the boundary conditions
 * 
 * Checks if the rank is at a global border and only applies BCs in tha case.
 * 
 * @param in    Input vector, also the output vector. the boundaries get overwritten.
 */
void SolverCG::ImposeBC(double* inout) {
    
    // BCs are applied at GLOBAL borders only, not local, indicated by absence of neighbour in that direction
    if (South == -2) { 
        for (int i = 0; i < NxLocal ; ++i) { 
            inout[IDX_LOCAL(i, 0)] = 0.0; 
        }
    }
    if (North == -2) {
        for (int i = 0; i < NxLocal ; ++i) {
            inout[IDX_LOCAL(i, NyLocal - 1)] = 0.0; 
        }
    }
    if (West == -2) { 
        for (int j = 0; j < NyLocal ; ++j) { 
            inout[IDX_LOCAL(0, j)] = 0.0; 
        }
    }
    if (East == -2) {
        for (int j = 0; j < NyLocal; ++j) { 
            inout[IDX_LOCAL(NxLocal - 1, j)] = 0.0; 
        }
    }
}

