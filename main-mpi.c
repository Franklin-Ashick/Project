#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <string.h> 
#include <mpi.h>

int readNumOfCoords(char *fileName);
double **readCoords(char *filename, int numOfCoords);
void *writeTourToFile(int *tour, int tourLength, char *filename);
double cheapestInsertionTSP(double **distMatrix, int *tour, int startVertex, int N);
double farthestInsertionTSP(double **distMatrix, int *tour, int startVertex, int N);
float nearestAdditionTSP(double **distMatrix, int *tour, int startVertex,int N);


// Struct to store tour and its cost
struct Tour {
  double cost;
  int tour[1000];
};

double **calculate_distance(double **arr, int length) {
    // Allocate a contiguous block of memory for the 2D array
    double (*dMatrix)[length] = malloc(length * length * sizeof(double));
    if (dMatrix == NULL) {
        perror("Memory Allocation Failed");
        exit(EXIT_FAILURE);
    }

    // Calculate the distance matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            dMatrix[i][j] = sqrt(((arr[i][0]  - arr[j][0]) * (arr[i][0]  - arr[j][0])) +
                                 ((arr[i][1] - arr[j][1]) * (arr[i][1] - arr[j][1])));
        }
    }

    // Cast the contiguous memory block to a pointer to an array of pointers
    // for compatibility with the rest of the program that expects a double**
    double **dMatrixPtrs = malloc(length * sizeof(double*));
    if (dMatrixPtrs == NULL) {
        perror("Memory Allocation Failed for dMatrixPtrs");
        free(dMatrix); // Don't forget to free the allocated block if the second allocation fails
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        dMatrixPtrs[i] = dMatrix[i];
    }

    return dMatrixPtrs;
}

int main(char argc, char *argv[]){

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // char *fileName = argv[1];
    char *fileName = "512_coords.coord";

    //char *outFileNameC = argv[2];
    //char *outFileNameF = argv[3];
    //char *outFileNameN = argv[4];

    // For time
    double tStart;
    double tEnd;

    int N = readNumOfCoords(fileName);

    double  **coords_twod_array = readCoords(fileName, N);
    double **distMatrix= calculate_distance(coords_twod_array, N);


    double localBestCostCI = DBL_MAX, localBestCostFI = DBL_MAX, localBestCostNA = DBL_MAX;
    int localBestTourCI[N + 1], localBestTourFI[N + 1], localBestTourNA[N + 1];
    int tour[N + 1], tourFI[N+1], tourNA[N+1];

    // Allocate memory to gather all best tours and costs on rank 0
    double *allBestCostsCI = NULL, *allBestCostsFI = NULL, *allBestCostsNA = NULL;
    // Array of arrays (each element is an int[N + 1])
    int (*allBestToursCI)[N + 1] = NULL;
    int (*allBestToursFI)[N + 1] = NULL;
    int (*allBestToursNA)[N + 1] = NULL;


    //Initialize all the global best costs and tours
    if (rank == 0) {
        allBestCostsCI = malloc(size * sizeof(double));
        allBestToursCI = malloc(size * sizeof(*allBestToursCI));
        if (allBestCostsCI == NULL || allBestToursCI == NULL) {
            perror("Memory Allocation Failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        allBestCostsFI = malloc(size * sizeof(double));
        allBestToursFI = malloc(size * sizeof(*allBestToursFI));
        if (allBestCostsFI == NULL || allBestToursFI == NULL) {
            perror("Memory Allocation Failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        allBestCostsNA = malloc(size * sizeof(double));
        allBestToursNA = malloc(size * sizeof(*allBestToursNA));
        if (allBestCostsNA == NULL || allBestToursNA == NULL) {
            perror("Memory Allocation Failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

    }

    // Start the timer
    tStart = MPI_Wtime();


    // Loop over each vertex as the starting point, distributed across ranks
    for (int i = rank; i < N; i += size) {

        // Cheapest Insertion TSP
        double costCI = cheapestInsertionTSP(distMatrix, tour, i, N);
        if (costCI < localBestCostCI) {
            localBestCostCI = costCI;
            memcpy(localBestTourCI, tour, (N + 1) * sizeof(int));
        }

        // Farthest Insertion TSP
       double costFI = farthestInsertionTSP(distMatrix, tourFI, i, N);
        if (costFI < localBestCostFI){
            localBestCostFI = costFI;
            memcpy(localBestTourFI, tourFI, (N + 1) * sizeof(int));
}
        // Nearest Addition TSP
        double costNA = nearestAdditionTSP(distMatrix, tour, i, N);
        if (costNA < localBestCostNA ) {
            localBestCostNA = costNA;
            memcpy(localBestTourNA, tour, (N + 1) * sizeof(int));
        }

}


    // End the timer
    tEnd = MPI_Wtime();

    // Gather all local best costs and tours at rank 0
    MPI_Gather(&localBestCostCI, 1, MPI_DOUBLE, allBestCostsCI, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(localBestTourCI, N + 1, MPI_INT, allBestToursCI, N + 1, MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Gather(&localBestCostFI, 1, MPI_DOUBLE, allBestCostsFI, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(localBestTourFI, N + 1, MPI_INT, allBestToursFI, N + 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(&localBestCostNA, 1, MPI_DOUBLE, allBestCostsNA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(localBestTourNA, N + 1, MPI_INT, allBestToursNA, N + 1, MPI_INT, 0, MPI_COMM_WORLD);



 if (rank == 0) {

        // Initialize the global best tour and cost
        double globalBestCostCI = DBL_MAX;
        int globalBestTourCI[N + 1];

        for (int i = 0; i < size; i++) {
            if (allBestCostsCI[i] < globalBestCostCI ) {
                globalBestCostCI = allBestCostsCI[i];
                memcpy(globalBestTourCI, allBestToursCI[i], (N + 1) * sizeof(int));
            }
        }

        double globalBestCostFI = DBL_MAX;
        int globalBestTourFI[N + 1];

        for (int i = 0; i < size; i++) {
            if (allBestCostsFI[i] < globalBestCostFI) {
                globalBestCostFI = allBestCostsFI[i];
                memcpy(globalBestTourFI, allBestToursFI[i], (N + 1) * sizeof(int));
            }
        }



        double globalBestCostNA = DBL_MAX;
        int globalBestTourNA[N + 1];

        for (int i = 0; i < size; i++) {
            if (allBestCostsNA[i] < globalBestCostNA){
                globalBestCostNA = allBestCostsNA[i];
                memcpy(globalBestTourNA, allBestToursNA[i], (N + 1) * sizeof(int));
            }
        }


       // Write the global best tours to files
       writeTourToFile(globalBestTourCI, N + 1, "best_ci.dat");
       writeTourToFile(globalBestTourFI, N + 1, "best_fi.dat");
       writeTourToFile(globalBestTourNA, N + 1, "best_na.dat");

        // Free the memory allocated for all best tours and costs
        free(allBestCostsCI);
        free(allBestToursCI);

        free(allBestCostsFI);
        free(allBestToursFI);

        free(allBestCostsNA);
        free(allBestToursNA);


     }

    free(distMatrix);
    free(coords_twod_array);

    MPI_Finalize();
    return 0;
}








