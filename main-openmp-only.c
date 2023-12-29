#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>  // for wallclock timing functions
#include <string.h>  


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
void getBestTourStartVertexZero(double cost,struct Tour *localBestTour, int *tourTSP,int N){
    if(cost  < localBestTour->cost) {
        localBestTour->cost = cost;
        memcpy(localBestTour->tour, tourTSP, (N + 1) * sizeof(int));
    }
}

void  getBestTourRest(double cost,struct Tour *localBestTour, int *tourTSP, int N){
    // Convert to integers after shifting decimal places
    long int_cost = (long)(cost*10000);
    long int_localBestCost = (long)(localBestTour->cost * 10000);

    if (int_cost < int_localBestCost) {
        localBestTour->cost = cost;
        memcpy(localBestTour->tour, tourTSP, (N + 1) * sizeof(int));
    }

}

int main(char argc, char *argv[]){

    char *fileName = argv[1];

    char *outFileNameC = argv[2];
    char *outFileNameF = argv[3];
    char *outFileNameN = argv[4];


    int N = readNumOfCoords(fileName);

    double  **coords_twod_array = readCoords(fileName, N);
    double **distMatrix= calculate_distance(coords_twod_array, N);


    struct Tour localBestTourC, localBestTourF,localBestTourN;
    //Initialising local best objects for c, f insertions and n addition
    localBestTourC.cost = DBL_MAX;
    localBestTourF.cost = DBL_MAX;
    localBestTourN.cost = DBL_MAX;
    memset(localBestTourC.tour, -1, sizeof(localBestTourC.tour)); // Initialize tour with -1
    memset(localBestTourF.tour, -1, sizeof(localBestTourF.tour)); // Initialize tour with -1
    memset(localBestTourN.tour, -1, sizeof(localBestTourN.tour)); // Initialize tour with -1

    // The tour array
    int *tour =(int *)malloc((N+1)*sizeof(int));

    // Run the TSP algorithms and find the local best tour for each process
    for (int i = 0; i < N; i++) {

    // Cheapest Insertion
    double costC =   cheapestInsertionTSP(distMatrix, tour, i, N);
    if(i==0){
         getBestTourStartVertexZero(costC,&localBestTourC, tour, N);
    } else {
         getBestTourRest(costC,&localBestTourC, tour, N);
    }

    // Farthest Insertion
    double costF = farthestInsertionTSP(distMatrix, tour, i, N);
    if(i==0){
         getBestTourStartVertexZero(costF,&localBestTourF, tour, N);
    } else {
         getBestTourRest(costF,&localBestTourF, tour, N);
    }

    // Nearest Addition
    double costN = nearestAdditionTSP(distMatrix, tour, i, N);
    if(i==0){
         getBestTourStartVertexZero(costN,&localBestTourN, tour, N);
    } else {
         getBestTourRest(costN,&localBestTourN, tour, N);
    }


 }

    writeTourToFile(localBestTourC.tour, N + 1, outFileNameC);
    writeTourToFile(localBestTourF.tour, N + 1, outFileNameF);
    writeTourToFile(localBestTourN.tour, N + 1, outFileNameN);



    free(distMatrix);
    free(tour);
    free(coords_twod_array);

 return 0;
}