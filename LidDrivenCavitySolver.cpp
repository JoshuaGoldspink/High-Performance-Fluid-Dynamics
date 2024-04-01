#include <iostream>
using namespace std;
#include "mpi.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"

/**
 * @brief Main function to solve the Lid Driven cavity Problem
 *
 * Main initialises MPI, creates a cartesian grid of ranks, sets up the problem and initialises variables, 
 * then calls various functions to solve the problem and write the output.
 * 
 * Uses conjugate gradient method to solve poisson equation, and explicit time integration.
 * Boundary conditions for the cavity problem are: 
 * all velocities are 0 at all edges except top edge, where velocity is equal to U
 * 
 * Uses MPI for coarse parallelisation and OpenMP for finer-grain parallelisation.
 * 
 *
 * @param   argc    Parameter to initialise MPI
 * @param   argv    Array of arrays containing info on all the command line inputs, which are the problem specifications
 * @return  0       Return 0 upon succesful completion of the main routine.

 */
int main(int argc, char **argv)
{   
    // Initialise MPI
    int world_rank;
    int world_size;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    
    // Variable to check size of MPI, and needed later in domain division.
    int n_rows = static_cast<int>(std::sqrt(world_size));
    
    // Make sure that the user input a P=p^2, perfect square number of ranks.
    if (n_rows * n_rows != world_size) {
        if (world_rank == 0) cerr << "Error: The number of MPI ranks " << world_size << " is not a perfect square.";
        MPI_Finalize();
        return 1; 
    }

    // Initialise cartesian coordinates
    MPI_Comm cart_comm;
    int dims[2] = {n_rows, n_rows};                 // p rows * p cols
    int periods[2] = {0, 0};                        // Non-periodic
    int reorder = 1;                                // reordering
    int coords[2];                               
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    // Assign the default values for the optional command prompt inputs
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(201),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(201),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.005),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(5),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(1000),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    // Help with problem inputs
    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    // Instantiate LDC class as solver
    LidDrivenCavity* solver = new LidDrivenCavity();
    
    // Set the problem specifications
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    
    // Set the problem MPI so it is accessible all throughout the class
    solver->SetMPI(cart_comm);
   
   
    // Compute each rank's local domain portion
    int rows_per_rank = vm["Nx"].as<int>() / n_rows;
    int extra_rows = vm["Nx"].as<int>() % n_rows;
    int cols_per_rank = vm["Ny"].as<int>() / n_rows;
    int extra_cols = vm["Ny"].as<int>() % n_rows;
    
    int startx = coords[0] * rows_per_rank + std::min(coords[0], extra_rows);       //< Start x-coordinate of local domain
    int endx = startx+ rows_per_rank - 1 + (coords[0] < extra_rows);                //< End x-coordinate of local domain
    int starty = coords[1] * cols_per_rank + std::min(coords[1], extra_cols);       //< Start y-coordinate of local domain
    int endy = starty + cols_per_rank - 1 + (coords[1] < extra_cols);               //< End y-coordinate of local domain
    
    // Logic for calculating the neighbours of each rank
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 1, -1, &north, &south); 
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    if (west != -2) startx -= 1;
    if (east != -2) endx += 1;
    if (north != -2) endy += 1;
    if (south != -2) starty -= 1;
    
    
    // Set Neighbours, so they are available in Solve() without passing it in each time. 
    solver->SetNeighbours(north, east, south, west);
    
    // Set the limits of the local domain, so they are accessible within the class
    solver->SetLimits(startx,endx,starty,endy);
   
    // Print the configuration once
    if (world_rank == 0) 
    solver->PrintConfiguration();

    // Initialise the problem
    solver->Initialise();

    // Write initial conditions
    solver->WriteSolution("ic.txt");

    // Solve the problem
    solver->Integrate();

    // Write the final solution to output file
    solver->WriteSolution("final.txt");
    
    // Finalize MPI
    MPI_Finalize();
    
    // Return 0 upon succesful code completion: no errors
	return 0;
}
