#define BOOST_TEST_MODULE AllTests
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <iostream>
#include "SolverCG.h"
#include "LidDrivenCavity.h"
#include <mpi.h>
#include <omp.h>

/*!*****************************************************************************
 * @brief  Define the indexing operations for global matrices, in row-major format
 * @note  indexing from (0,0) in bottom left corner.
 *******************************************************************************/
#define IDX(i,j) ((j)*Nx + (i))



/**
 * @struct GlobalMPIFixture
 * @brief Initializes and finalizes the MPI environment.
 *
 * 
 * Needed for the BOOST unit tests to function with mpi correctly.
 * Makes MPI initialise and shut down correclty for the unit tests.
 * 
 */
struct GlobalMPIFixture {
    
    /**
     * @brief Constructor for GlobalMPIFixture.
     * Initializes MPI with MPI_Init.
     */
    GlobalMPIFixture() {
        MPI_Init(NULL, NULL);
    }
    
    /**
     * @brief Destructor for GlobalMPIFixture.
     * Terminates MPI with MPI_Finalize().
     */
    ~GlobalMPIFixture() {
        MPI_Finalize();
    }
};

BOOST_GLOBAL_FIXTURE(GlobalMPIFixture);


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
void printMatrixTest(const double* array, int Nx, int Ny) {
    for (int j = Ny - 1; j >= 0; --j) {
        for (int i = 0; i < Nx; ++i) {
            int index = j * Nx + i;
            // Use std::fixed and std::setprecision to control the number format
            // std::setw(10) ensures that each number takes up 10 characters' space for alignment
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << array[index] << " ";
        }
        std::cout << std::endl; // Newline after each row
    }
}


/**
 * @brief Calculates analytical solution for Solver.
 *
 * Uses discretised version of sin(pi k x) sin (pi l y)
 * 
 * @param   i       x location index
 * @param   j       y location index
 * @param   dx      discretisation in y direction
 * @param   dxy     discretisation in y direction
 * @param   k       coefficient k
 * @param   l       coefficient l
 */
double analytical_solution(int i, int j, double dx, double dy, int k, int l) {
    return sin(M_PI * k * i * dx) * sin(M_PI * l * j * dy);
}


/**
 * @brief Tests the MPI access of BOOST tests, and initialisaion of MPI.
 *
 * Used to debug the unit tests themselves and to make sure 
 * MPI is available in the unit tests.
 * 
 */
BOOST_AUTO_TEST_CASE(MPITest) {
    int world_rank;
    int world_size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    
    if (world_rank == 0) cout << "1. MPI Initialisation test" <<  endl;
    
    BOOST_CHECK( world_size != 0);
}


/**
 * @brief tests SolverCG class
 *
 * Uses the initial comndition v = - pi^2 (k^2 + l^2) sin(pi k x) sin(pi l y)
 * For simplicity, calls a new AdvanceTest() functionl, and sets the final time to one delta_t,
 * so that it just solves the poisson and nothing else.
 * 
 */
BOOST_AUTO_TEST_CASE(SinusoidalTest) {
    int world_rank;
    int world_size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    
    if (world_rank == 0) cout << "2. Testing sinusoidal input for Poisson Equation" << endl;
    
    const int Nx = 20; // Grid size in x-direction
    const int Ny = 20; // Grid size in y-direction
    const double dx = 1.0 / (Nx - 1);
    const double dy = 1.0 / (Ny - 1);
    const int k = 3;
    const int l = 3;
    const double tolerance = 0.01;
    const int Npts = Nx * Ny;
    double* s_analytical = new double[Npts]; //stores analytical answer for test case
    double* v = new double[Npts]; 
    double* s = new double[Npts]; 
    
    
    //Set analytical solution
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            s_analytical[IDX(i,j)] = -analytical_solution(i, j, dx, dy, k, l);
        }
    }
    
    
    
    
    int n_rows = sqrt(world_size);
    MPI_Comm cart_comm;
    int dims[2] = {n_rows, n_rows};                 // p rows * p cols
    int periods[2] = {0, 0};                        // Non-periodic
    int reorder = 1;                                // reordering
    int coords[2];                               
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);


    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(1.0, 1.0);
    solver->SetGridSize(Nx, Ny);
    solver->SetTimeStep(0.001);
    solver->SetFinalTime(0.001);
    solver->SetReynoldsNumber(10);
    solver->SetMPI(cart_comm);
   
    int rows_per_rank = Nx / n_rows;
    int extra_rows = Nx % n_rows;
    int cols_per_rank = Ny / n_rows;
    int extra_cols = Ny % n_rows;
    
    int startx = coords[0] * rows_per_rank + std::min(coords[0], extra_rows);
    int endx = startx+ rows_per_rank - 1 + (coords[0] < extra_rows);
    int starty = coords[1] * cols_per_rank + std::min(coords[1], extra_cols);
    int endy = starty + cols_per_rank - 1 + (coords[1] < extra_cols);
    

    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 1, -1, &north, &south); 
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    if (west != -2) startx -= 1;
    if (east != -2) endx += 1;
    if (north != -2) endy += 1;
    if (south != -2) starty -= 1;
    
    
    //Set Neighbours, so they are available in Solve() without passing it in each time. 
    solver->SetNeighbours(north, east, south, west);
    
    solver->SetLimits(startx,endx,starty,endy);
   
    solver->Initialise();

    solver->AdvanceTest();
    
    s = solver->getGlobalSolution();
    
    
    //Calculate mean squared error: simple and element-wise comparison
    double mse = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double error = s[IDX(i,j)] - s_analytical[IDX(i,j)];
            mse += error * error; // Squaring the error
        }
    } 

    mse /= (Nx * Ny); // Divide by the number of points to get the mean

    
    // Clean up
    delete[] v;
    delete[] s;
    delete[] s_analytical;
    
    if (world_rank == 0) cout << "Testing complete for SolverCG.cpp: mse was " << mse << endl;
    
    BOOST_CHECK(mse < tolerance*tolerance); //error squared is tolerance squared
   
}


/**
 * @brief tests LidDrivenCavity class
 *
 * 
 * Simply initialises the LDC class and makes sure nothing goes wrong. 
 * Checks to see if LDC succesfully returns a streamfunction matrix.
 * 
 */
BOOST_AUTO_TEST_CASE(LDCTest) {
    int world_rank;
    int world_size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    
    if (world_rank == 0) cout << "3. Testing LDC class" << endl;
    
    const int Nx = 15; // Grid size in x-direction
    const int Ny = 15; // Grid size in y-direction
    const int Npts = Nx * Ny;
    double* s = new double[Npts]; 
    
    
    
    
    int n_rows = sqrt(world_size);
    
    MPI_Comm cart_comm;
    int dims[2] = {n_rows, n_rows};                 
    int periods[2] = {0, 0};                        // Non-periodic
    int reorder = 1;                                // reordering
    int coords[2];                               
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);


    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(1.0, 1.0);
    solver->SetGridSize(Nx, Ny);
    solver->SetTimeStep(0.001);
    solver->SetFinalTime(0.001);
    solver->SetReynoldsNumber(10);
    solver->SetMPI(cart_comm);
   
    int rows_per_rank = Nx / n_rows;
    int extra_rows = Nx % n_rows;
    int cols_per_rank = Ny / n_rows;
    int extra_cols = Ny % n_rows;
    
    int startx = coords[0] * rows_per_rank + std::min(coords[0], extra_rows);
    int endx = startx+ rows_per_rank - 1 + (coords[0] < extra_rows);
    int starty = coords[1] * cols_per_rank + std::min(coords[1], extra_cols);
    int endy = starty + cols_per_rank - 1 + (coords[1] < extra_cols);
    

    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 1, -1, &north, &south); 
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    if (west != -2) startx -= 1;
    if (east != -2) endx += 1;
    if (north != -2) endy += 1;
    if (south != -2) starty -= 1;
    
    
    //Set Neighbours, so they are available in Solve() without passing it in each time. 
    solver->SetNeighbours(north, east, south, west);
    
    solver->SetLimits(startx,endx,starty,endy);
   
    solver->Initialise();

    solver->Integrate();
    
    s = solver->getGlobalSolution();
    
    if (world_rank == 0) cout << "Testing complete for LDC Class" << endl;
    BOOST_CHECK(s);
    
    // Clean up
    delete[] s;
}




