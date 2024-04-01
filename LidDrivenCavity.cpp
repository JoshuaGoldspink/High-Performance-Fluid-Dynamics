#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>

using namespace std;



/*!*****************************************************************************
 * @brief  Define the indexing operations for both local and global matrices, in row-major format
 * @note  indexing from (0,0) in bottom left corner.
 *******************************************************************************/
#define IDX(I,J) ((J)*Nx + (I))
#define IDX_LOCAL(I,J) ((J)*NxLocal + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Constructor for the LDC class. 
 * 
 * The setup is handled in other functions, so this constructor is empty for simplicity.
 */
LidDrivenCavity::LidDrivenCavity()
{
}

/**
 * @brief Destructor for the LDC class.
 *
 * Executed after the code is finished using the LDC class, and simply calls CleanUp() 
 * to clear all the vairbales and de-allocating the dynamic memory.
 *
 */
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/**
 * @brief Sets global size of problem domain.
 * 
 * 
 *
 * @param   xlen     Length of the global domain in the x direction, in meters.
 * @param   ylen     Length of the global domain in the y direction, in meters.

 */
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

/**
 * @brief Sets size of global problem grid.
 *
 * @param   nx       Number of grid points in the x direction.
 * @param   ny       Number of grid points in the y direction.
 */
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}

/**
 * @brief Sets the problem time-step.
 *
 * @param   deltat      Value of the time-step.
 */
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/**
 * @brief Sets the Final time.
 * 
 * This is the total amount of time over which we simulate the evolution of 
 * the lid driven flow problem. 
 *
 * @param   finalt      Stopping time of the problem.
 */
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/**
 * @brief Sets Reynolds number.
 *
 * @param   re      Desire problem Reynolds number
 */
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

/**
 * @brief Sets up the MPI.
 *
 * Makes the MP*I comunicator available to the entire LDC class, necessary
 * for the boundary data sharing.
 * 
 * @param   comm    MPI comunicator to be set for the LDc class.
 */
void LidDrivenCavity::SetMPI(MPI_Comm pComm) {
    comm = pComm;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);
}

/**
 * @brief Sets each ranks neighbours.
 *
 * Each value is an integer corresponding to the rank of the neighbour in that direction. 
 * If the rank is at a global border and has no neighbour in a given direction, the vlaue will be -2.
 * 
 * @param   north   Ranks North neighbour
 * @param   east    Ranks east neighbour
 * @param   south   Ranks south neighbour
 * @param   west    Ranks west neighbour
 */
void LidDrivenCavity::SetNeighbours(int north, int east, int south, int west) {
    this->North = north;
    this->East = east;
    this->South = south;
    this->West = west;
}

/**
 * @brief Sets the ranks global coordinates.
 *
 * Includes the ghost cells, gives the indexes of the start and end x and y position, inclusive.
 * 
 * @param   startx   Starting x coordinate
 * @param   endx   Ending y coordinate
 * @param   starty   Starting x coordinate
 * @param   endy   Ending y coordinate
 */
void LidDrivenCavity::SetLimits(int startx, int endx, int starty, int endy){
    this->Startx = startx;
    this->Endx = endx;
    this->Starty = starty;
    this->Endy = endy;
    
    //Set local dimensions
    this->NxLocal = endx - startx + 1;
    this->NyLocal = endy - starty + 1;
    
}

/**
 * @brief Initialises the LDC class.
 *
 * Cleans any variables, then creates dynamic memory for vorticity, streamfunction, temporary variables, and instantiates 
 * the solveCG class.
 * Also calculates the size of the local problem to initialise the size of the local variables.
 */
void LidDrivenCavity::Initialise()
{
    CleanUp();
    int NxLocal = Endx - Startx + 1;
    int NyLocal = Endy - Starty + 1;
    int n_local = NxLocal * NyLocal;
    
    v   = new double[n_local]();
    vnew = new double[n_local]();
    s   = new double[n_local]();
    tmp = new double[n_local]();
    cg  = new SolverCG(Nx, Ny, dx, dy, Startx, Endx, Starty, Endy, comm, North, East, South, West);
}

/**
 * @brief Solves for vorticity and streamfunction.
 *
 * Iterates through every time step and uses an explicit integration method to advance by delta_t each time. 
 * Print the convergence to the terminal at each time step.
 * 
 */
void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);
    for (int t = 0; t < NSteps; ++t)
    {
        if (world_rank == 0 ) 
        std::cout << "Step: " << setw(8) << t << "  Time: " << setw(8) << t*dt << std::endl;
        Advance();
    }
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
void printMatrix(const double* array, int Nx, int Ny) {
    for (int j = Ny - 1; j >= 0; --j) {
        for (int i = 0; i < Nx; ++i) {
            int index = j * Nx + i;
            std::cout << std::setw(4) << std::fixed << std::setprecision(2) << array[index] << " ";
        }
        std::cout << std::endl; // Newline after each row
    }
}

/**
 * @brief Removes ghost cells.
 *
 * Used for condensing the outputs of each rank into a global solution. 
 * 
 * @param   s               Global solution matrix, of size Nx*Ny.
 * @param   s_clean         Local solution matrix without padding. 
 * @param   Startx          Starting x coordinate
 * @param   Endx            Ending y coordinate
 * @param   Starty          Starting x coordinate
 * @param   Endy            Ending y coordinate
 * @param   Startx_nb       Starting x coordinate without padding
 * @param   Endx_nb         Ending y coordinate without padding
 * @param   Starty_nb       Starting x coordinate without padding
 * @param   Endy_nb         Ending y coordinate without padding
 * 
 */
void LidDrivenCavity::copyNonBorderValues(double* s, double* s_clean, int Startx, int Endx, int Starty, int Endy, int Startx_nb, int Starty_nb, int Endx_nb, int Endy_nb) {
    int NxLocal = Endx - Startx + 1;
    int NxLocal_clean = Endx_nb - Startx_nb + 1;

    for (int y = Starty_nb; y <= Endy_nb; ++y) {
        for (int x = Startx_nb; x <= Endx_nb; ++x) {
            int index_s = (y - Starty) * NxLocal + (x - Startx);
            int index_s_clean = (y - Starty_nb) * NxLocal_clean + (x - Startx_nb);
            s_clean[index_s_clean] = s[index_s];
        }
    }
}

/**
 * @brief Calculates non-padded dimensions.
 *
 * Used for condensing each ranks output into a global output,
 * gathers data needed later for MPI_Gatherv.
 * 
 * 
 * @param   allStartX       vector containing the startx coordinates for each rank
 * @param   allEndX         vector containing the endx coordinates for each rank
 * @param   allStartY       vector containing the starty coordinates for each rank
 * @param   allEndY         vector containing the endy coordinates for each rank
 * 
 */
void LidDrivenCavity::gatherMatrixDimensions(std::vector<int>& allStartX, std::vector<int>& allEndX, std::vector<int>& allStartY, std::vector<int>& allEndY
                , int Startx_nb, int Endx_nb, int Starty_nb, int Endy_nb) {
                    
    //Gather data from each rank into the vectors, needed later for MPI_gatherv
    MPI_Gather(&Startx_nb, 1, MPI_INT, allStartX.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&Endx_nb, 1, MPI_INT, allEndX.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&Starty_nb, 1, MPI_INT, allStartY.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&Endy_nb, 1, MPI_INT, allEndY.data(), 1, MPI_INT, 0, comm);
}

/**
 * @brief Assembles global solution matrix.
 * 
 * Uses the non-padded matrices from each rank to condense them into the global
 * solution matrix by putting them each in the right section of the domain. 
 * 
 * @param clean_b           Pointer to the non-padded local matrix data for the current rank.
 * @param globalMatrix      Pointer to the final matrix (global).
 * @param Startx_nb         The startx without padding of this ranks local matrix.
 * @param Endx_nb           The endx without padding of this ranks local matrix.
 * @param Starty_nb         The starty without padding of this ranks local matrix.
 * @param Endy_nb           The endy without padding of this ranks local matrix.
 */
void LidDrivenCavity::assembleGlobalMatrix(double* clean_b, double* globalMatrix, int Startx_nb, int Endx_nb, int Starty_nb, int Endy_nb){
    int localMatrixSize = (Endx_nb - Startx_nb + 1) * (Endy_nb - Starty_nb + 1);
    
    // Create vectors containing the start and end coordinates of each rank, which we need to calculate 
    // The displacements and counts for each rank.
    std::vector<int> allStartX(world_size), allEndX(world_size), allStartY(world_size), allEndY(world_size);
    gatherMatrixDimensions(allStartX, allEndX, allStartY, allEndY, Startx_nb, Endx_nb, Starty_nb, Endy_nb);
    
    // Calculate counts and displacements
    std::vector<int> recvCounts(world_size), displs(world_size);
    if (world_rank == 0) {
            int totalSize = 0;
            for (int i = 0; i < world_size; ++i) {
                recvCounts[i] = (allEndX[i] - allStartX[i] + 1) * (allEndY[i] - allStartY[i] + 1);
                displs[i] = totalSize;
                totalSize += recvCounts[i];
            }
        }

    // Allocate a buffer for the global data if on rank 0
    double* globalData = nullptr;
    if (world_rank == 0) {
        globalData = new double[Nx * Ny]();
    }

    // Gather all the local matrices into the global data buffer on rank 0
    MPI_Gatherv(clean_b, localMatrixSize, MPI_DOUBLE, 
                globalData, recvCounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // Place the received data into their correct positions in the global matrix
    if (world_rank == 0) {
        for (int rank = 0; rank < world_size; ++rank) {
            int width = allEndX[rank] - allStartX[rank] + 1;
            for (int i = 0; i < recvCounts[rank]; ++i) {
                int row = i / width;
                int col = i % width;
                int globalRow = allStartY[rank] + row;
                int globalCol = allStartX[rank] + col;
                int globalIdx = globalRow * Nx + globalCol;
                globalMatrix[globalIdx] = globalData[displs[rank] + i];
            }
        }
    }
    //Clean up, de-allocate
    if (globalData) delete[] globalData;
}

/**
 * @brief Get the final streamfunction for the unit tests.
 *
 * Simply condenses the vlaues together but also returns the array itself, instead of just overwriting another array,
 * because we need to return values for the unit test to check. 
 * 
 * @return  s   pointer to the final solution matrix for the streamfunction.
 */
double* LidDrivenCavity::getGlobalSolution() {
    
    int Startx_nb = Startx, Endx_nb = Endx, Starty_nb = Starty, Endy_nb = Endy;
    if (North != -2) Endy_nb = Endy - 1;
    if (South != -2) Starty_nb = Starty + 1;
    if (East != -2) Endx_nb = Endx - 1;
    if (West != -2) Startx_nb = Startx + 1;
    
    
    // Calculate size of local matrix without borders
    int Nx_nb = Endx_nb - Startx_nb + 1;
    int Ny_nb = Endy_nb - Starty_nb + 1;
    
    // Initialise clean versions of local matrices
    double* s_clean = new double[Nx_nb * Ny_nb]; 
    double* s_return = new double[Nx*Ny];
    
    // Remove ghost cells before recombining into final output
    copyNonBorderValues(s, s_clean, Startx, Endx, Starty, Endy, Startx_nb, Starty_nb, Endx_nb, Endy_nb); 
    
    // Assemble local matrices into global
    assembleGlobalMatrix(s_clean, s_return, Startx_nb, Endx_nb, Starty_nb, Endy_nb);
   
    MPI_Bcast(s_return, Nx * Ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
    // Clean local variables holding the non border values.
    delete[] s_clean;
    
    return s_return;
}

/**
 * @brief Write the global solution to output file.
 *
 * First, calls the functions to assemble the final matrix from local ones. Then write the output.
 *
 * @param   file    desired output file
 */
void LidDrivenCavity::WriteSolution(std::string file)
{   
    
    // First, remove the border cells: Startx_nb is no-border.
    
    int Startx_nb = Startx, Endx_nb = Endx, Starty_nb = Starty, Endy_nb = Endy;
    if (North != -2) Endy_nb = Endy - 1;
    if (South != -2) Starty_nb = Starty + 1;
    if (East != -2) Endx_nb = Endx - 1;
    if (West != -2) Startx_nb = Startx + 1;
    
    
    // Calculate size of local matrix without borders
    int Nx_nb = Endx_nb - Startx_nb + 1;
    int Ny_nb = Endy_nb - Starty_nb + 1;
    
    // Initialise clean versions of local matrices
    double* s_clean = new double[Nx_nb * Ny_nb]; 
    double* v_clean = new double[Nx_nb * Ny_nb]; 
    
    // Remove ghost cells befoe recombininb into final output
    copyNonBorderValues(s, s_clean, Startx, Endx, Starty, Endy, Startx_nb, Starty_nb, Endx_nb, Endy_nb); 
    copyNonBorderValues(v, v_clean, Startx, Endx, Starty, Endy, Startx_nb, Starty_nb, Endx_nb, Endy_nb); 
    
    // Allocate final matrix: reinitialise s and v but as global matrices, not local
    double* s = new double[Nx*Ny];
    double* v = new double[Nx*Ny];
    assembleGlobalMatrix(s_clean, s, Startx_nb, Endx_nb, Starty_nb, Endy_nb);
    assembleGlobalMatrix(v_clean, v, Startx_nb, Endx_nb, Starty_nb, Endy_nb);
    
    // Clean local variables holding the non border values.
    delete[] s_clean;
    delete[] v_clean;
    
    // Code for printing output
    /*
    if (world_rank ==0){
    cout << "full matrix s " << endl;
    printMatrix(s, Nx, Ny);
    cout << "full matrix v: " << endl;
    printMatrix(v, Nx, Ny);
    }*/
    
    
    // Keep the original code for writing the output to file. 
    // But only write from rank 0 which holds the global matrices v and s.
    
    if (world_rank == 0) {
        double* u0 = new double[Nx*Ny]();
        double* u1 = new double[Nx*Ny]();
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;
                u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;
            }
        }
        for (int i = 0; i < Nx; ++i) {
            u0[IDX(i,Ny-1)] = U;
        }

        std::ofstream f(file.c_str());
        std::cout << "Writing file " << file << std::endl;
        int k = 0;
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                k = IDX(i, j);
                f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
                  << " " << u0[k] << " " << u1[k] << std::endl;
            }
            f << std::endl;
        }
        f.close();
        delete[] u0;
        delete[] u1;
    }

}

/**
 * @brief Prints problem configuration.
 *
 */
void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

/**
 * @brief Cleans variables and memory.
 *
 * De-allocates the dynamic memory for the variables v, vnew, s, tmp, and cg.
 * 
 */
void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete cg;
    }
}

/**
 * @brief Updates Dx & Dy.
 *
 * Updates discretisation of the problem, constant across the ranks. 
 * Also calculates the total number of points.
 */
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx * Ny;
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
void exchangeBoundaryData(double* s, int NxLocal, int NyLocal, int north, int south, int east, int west) {
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
 * @brief Divides global matrix across the ranks.
 *
 * Used for testing cases. Divides a global matrixes into portions across each rank.
 * 
 * 
 * @param   b               pointer to global matrix we wish to divide.
 * @param   local_b         pointer to local matrix we wish to fill.
 * @param   Nx              size of b in x.
 * @param   Ny              size of b in y.
 * @param   Startx          Starting x coordinate
 * @param   Starty          Starting x coordinate
 * @param   Endx            Ending y coordinate
 * @param   Endy            Ending y coordinate
 * 
 */
void segmentMatrix(const double* b, double* local_b, int Nx, int Ny, int Startx, int Starty, int Endx, int Endy) {
    int LocalNx = Endx - Startx + 1;
    int LocalNy = Endy - Starty + 1;

    for (int i = 0; i < LocalNx; ++i) {
        for (int j = 0; j < LocalNy; ++j) {
            // Calculate global indices based on the bottom-left as (0,0)
            int globalIndex = ((Starty + j) * Nx) + (Startx + i);
            // Calculate local index based on local matrix dimensions
            int localIndex = (j * LocalNx) + i;

            local_b[localIndex] = b[globalIndex];
        }
    }
}

/**
 * @brief Integrates the problem to the next time-step
 *
 * Uses explicit integration. 
 * Starts by calculating boundary node vorticity, then interior
 * vorticity, then time advances vorticity, then solves Poisson
 * equation with solverCG to find streamfunction at time t+delta_t.

 */
void LidDrivenCavity::Advance()
{
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;     
    
    
    // Boundary node vorticity
    #pragma omp parallel for
    for (int i = 1; i < NxLocal-1; ++i) {
        if (South == -2) {
            // bottom physical boundary
            v[IDX_LOCAL(i,0)] = 2.0 * dy2i * (s[IDX_LOCAL(i,0)] - s[IDX_LOCAL(i,1)]);
        }
        if (North == -2) {
            // top physical boundary
            v[IDX_LOCAL(i,NyLocal-1)] = 2.0 * dy2i * (s[IDX_LOCAL(i,NyLocal-1)] - s[IDX_LOCAL(i,NyLocal-2)])
                                        - 2.0 * dyi*U;
        }
    }
    
    #pragma omp parallel for
    for (int j = 1; j < NyLocal-1; ++j) {
        if (West == -2) {
            // left physical boundary
            v[IDX_LOCAL(0,j)] = 2.0 * dx2i * (s[IDX_LOCAL(0,j)] - s[IDX_LOCAL(1,j)]);
        }
        if (East == -2) {
            // right physical boundary
            v[IDX_LOCAL(NxLocal-1,j)] = 2.0 * dx2i * (s[IDX_LOCAL(NxLocal-1,j)] - s[IDX_LOCAL(NxLocal-2,j)]);
        }
    }
    
    // Make sure local boundaries are up to date.
    // Both v and s are needed for following computations
    exchangeBoundaryData(v, NxLocal, NyLocal, North, South, East, West);
    exchangeBoundaryData(s, NxLocal, NyLocal, North, South, East, West);
    
    
   
    // Compute interior vorticity
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < NyLocal - 1; ++j){              //Switch i and j for more efficient memory usage
         for (int i = 1; i < NxLocal - 1; ++i){
            v[IDX_LOCAL(i,j)] = dx2i*(
                    2.0 * s[IDX_LOCAL(i,j)] - s[IDX_LOCAL(i+1,j)] - s[IDX_LOCAL(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX_LOCAL(i,j)] - s[IDX_LOCAL(i,j+1)] - s[IDX_LOCAL(i,j-1)]);
        }
    }
    
    // Re-update boundaries because we updated v
    exchangeBoundaryData(v, NxLocal, NyLocal, North, South, East, West);
    

    // Time advance vorticity
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < NyLocal - 1; ++j) {             //Switch i and j for more efficient memory usage
         for (int i = 1; i < NxLocal - 1; ++i) {
            vnew[IDX_LOCAL(i,j)] = v[IDX_LOCAL(i,j)] + dt*(
                ( (s[IDX_LOCAL(i+1,j)] - s[IDX_LOCAL(i-1,j)]) * 0.5 * dxi
                 *(v[IDX_LOCAL(i,j+1)] - v[IDX_LOCAL(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX_LOCAL(i,j+1)] - s[IDX_LOCAL(i,j-1)]) * 0.5 * dyi
                 *(v[IDX_LOCAL(i+1,j)] - v[IDX_LOCAL(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX_LOCAL(i+1,j)] - 2.0 * v[IDX_LOCAL(i,j)] + v[IDX_LOCAL(i-1,j)])*dx2i
              + nu * (v[IDX_LOCAL(i,j+1)] - 2.0 * v[IDX_LOCAL(i,j)] + v[IDX_LOCAL(i,j-1)])*dy2i);
        }
    }
    
    // Update again because we changed vnew
    exchangeBoundaryData(vnew, NxLocal, NyLocal, North, South, East, West);
    
    

    // Solve Poisson problem
    cg->Solve(vnew, s);
}

/**
 * @brief Does not time-advance, just calls the Solve() function once.
 *
 * Used to test the poisson solver by calling the solve function on a pre-determined initial
 * condition which has an analytical solution. 
 * 
 * @note v_initial = -pi^2 (k^2 + l^2) sin(pi k x) sin (pi l y)

 */
void LidDrivenCavity::AdvanceTest()
{
    
    // Test case:
    
    int k = 3;
    int l = 3;
    double* v_global = new double[Nx*Ny];
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            v_global[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    
    
    segmentMatrix(v_global, v, Nx, Ny, Startx, Starty, Endx, Endy);
    
    delete[] v_global;
     
    // Solve Poisson problem
    cg->Solve(v, s);
}

