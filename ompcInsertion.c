#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include<omp.h>
#include <string.h>  // This includes the declaration of memcpy


// Cheapest Insertion TSP Algorithm
double cheapestInsertionTSP(double **distMatrix, int *tour, int startVertex, int N) {
    bool *visited = (bool *)calloc(N, sizeof(bool));

    int tourSize = 1;
    tour[0] = startVertex;
    visited[startVertex] = true;

    while (tourSize < N) {
        double minCost = DBL_MAX;
        int minPos = -1;
        int minVertex = -1;
        // Parallel for loop to find the cheapest unvisited vertex from the tour
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (!visited[i]) {
                double localMinCost = DBL_MAX;
                int localMinPos = -1;
                int localMinVertex = i;

                for (int j = 0; j < tourSize; j++) {
                    int k = (j + 1) % tourSize;
                    double cost = distMatrix[tour[j]][i] + distMatrix[i][tour[k]] - distMatrix[tour[j]][tour[k]];    // Calculate the insertion cost

                    if (cost < localMinCost) {
                       localMinCost = cost;
                        localMinPos = j + 1; // Insert after position j
                    }
                }
            // Critical section to update the global minimum
                #pragma omp critical
                {
                    if (localMinCost < minCost) {
                        minCost = localMinCost;
                        minVertex = localMinVertex;
                        minPos = localMinPos;
                    }
                }
            }
        }
        //Insert the vertex into the tour at the specified position
        for (int i = tourSize; i > minPos; i--) {
          tour[i] = tour[i - 1];
        }
        tour[minPos] = minVertex;


        tourSize++;

        visited[minVertex] = true;
    }
// Add the logic to close the tour by returning to the start vertex
    tour[N] = tour[0];

    // Compute the total cost of the tour
    double tourCost = 0.0;
    for (int i = 0; i < N; i++) {
        tourCost += distMatrix[tour[i]][tour[(i + 1) % (N + 1)]];
    }

    return tourCost; // Return the total cost of the tour
}