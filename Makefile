CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall -O2 -fopenmp
LIBS = -lboost_unit_test_framework -lblas -lboost_program_options 

# Define targets
MAIN = LDC
TEST = AllTests

# Main program source and object files
MAIN_SRC = SolverCG.cpp LidDrivenCavitySolver.cpp LidDrivenCavity.cpp
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# Test program source and object files
TEST_SRC = AllTests.cpp SolverCG.cpp LidDrivenCavity.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)

# Default target
all: $(MAIN)

# Main program
$(MAIN): $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)


# Test program
$(TEST): $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Accept command line arguments
NP ?= 1
CMD_ARGS := $(if $(Lx),--Lx=$(Lx),)
CMD_ARGS += $(if $(Ly), --Ly=$(Ly),)
CMD_ARGS += $(if $(Nx), --Nx=$(Nx),)
CMD_ARGS += $(if $(Ny), --Ny=$(Ny),)
CMD_ARGS += $(if $(dt), --dt=$(dt),)
CMD_ARGS += $(if $(T), --T=$(T),)
CMD_ARGS += $(if $(Re), --Re=$(Re),)
CMD_ARGS += $(if $(p), --p=$(p),)

# Unit tests
unittests: $(TEST)
	mpiexec -np $(NP) ./$(TEST) $(CMD_ARGS)

# Solver
solver: $(MAIN)
#	mpiexec -np $(NP) ./$(MAIN) $(CMD_ARGS)
	mpirun --bind-to none -np $(NP) ./$(MAIN) $(CMD_ARGS)

# Clean up
clean:
	rm -f $(MAIN) $(TEST) $(MAIN_OBJ) $(TEST_OBJ)

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

doc: 
	doxygen doxyfile

.PHONY: all clean unittests run
