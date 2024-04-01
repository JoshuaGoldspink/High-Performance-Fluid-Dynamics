#pragma once
#include <mpi.h>
#include <omp.h>
#include <string>
using namespace std;

class SolverCG;



/**
 * @class       LidDrivenCavity
 * @brief       LDC class represents the overall lid driven cavity problem. It contains functionality
 *              to set-up the problem, initialise values, solve the problem, then write the solution to a file. 
 * 
 * 
 * @note        Requires MPI and OpenMP to be run in parallel.
 */
class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);                   ///< Sets the length of the domain in the x and y directions.
    void SetGridSize(int nx, int ny);                               ///< Sets the grid discretisation in the x and y directions.
    void SetTimeStep(double deltat);                                ///< Sets the time increment for integration.
    void SetFinalTime(double finalt);                               ///< Sets the Final time.
    void SetReynoldsNumber(double Re);                              ///< Sets the Reynolds number of the flow.
    void SetMPI(MPI_Comm pcomm);                                    ///< Sets the MPI Communicator for the class.
    void SetNeighbours(int north, int east, int south, int west);   ///< Sets the neighbours for the current rank.
    void SetLimits(int startx, int endx, int starty, int endy);     ///< Sets the coordinates of the local domain within the global domain.
    
    
    void Initialise();                          ///< Initialises the class.
    void Integrate();                           ///< Time advances acorss every timestep.
    void WriteSolution(std::string file);       ///< Write the initial conditions to file.
    void PrintConfiguration();                  ///< Prints the configuration.
    
    // Combines the local solutions into one
    void assembleGlobalMatrix(double* clean_b, double* globalMatrix, int Startx_nb, int Endx_nb, int Starty_nb, int Endy_nb);
    
    // Gathers the local dimensions of each rank onto rank 0
    void gatherMatrixDimensions(std::vector<int>& allStartX, std::vector<int>& allEndX, std::vector<int>& allStartY, std::vector<int>& allEndY
    , int Startx_nb, int Endx_nb, int Starty_nb, int Endy_nb);
    
    // Removes the borders from local solution.
    void copyNonBorderValues(double* s, double* s_clean, int Startx, int Endx, int Starty, int Endy, int Startx_nb, int Starty_nb, int Endx_nb, int Endy_nb);
    
    // Tests the poisson solver.
    void AdvanceTest();
    
    // Gets the globalSolution for the unit tests.
    double* getGlobalSolution();
    

private:
    double* v   = nullptr;      ///< Vorticity matrix.
    double* vnew = nullptr;     ///< New vorticity matrix after time advancement.
    double* s   = nullptr;      ///< Streamfunction matrix.
    double* tmp = nullptr;      ///< Temporary matrix.

    double dt  = 0.005;         ///< Timestep.
    double T    = 0.5;          ///< Final time.
    double dx;                  ///< Grid discretisation in the x-direction.
    double dy;                  ///< Grid discretisation in the x-direction.
    int  Nx   = 201;            ///< Number of points in the x-direction.
    int  Ny   = 201;            ///< Number of points in the y-direction.
    int NxLocal;                ///< Number of points in the x-direction of the local domain.
    int NyLocal;                ///< Number of points in the Y-direction of the local domain.
    int    Npts = 40401;        ///< Total number of points in the global domain.
    double Lx   = 1.0;          ///< Length of the grid in the x-direction.
    double Ly   = 1.0;          ///< Number of grid in the y-direction.
    double Re   = 1000;           ///< Flow Reynolds number.
    double U    = 1.0;          ///< Boundary fluid velocity.
    double nu   = 0.001;          ///< Fluid viscosity.

    MPI_Comm comm;              ///< MPI communicator.
    int world_rank = -1;        ///< Rank within the global communicator.
    int world_size = 1;         ///< Total number of ranks.
    
    int North, East, South, West;       ///< Integers of corresponding neighours ranks.
    int Startx, Endx, Starty, Endy;     ///< Coordinates defining the local domain within global domain.
    SolverCG* cg = nullptr;             ///< Solver pointer.
    
    // Clears memory
    void CleanUp(); 

    // Updates discretisation
    void UpdateDxDy();
    
    // Time-advances problem by delta_t
    void Advance();
};

