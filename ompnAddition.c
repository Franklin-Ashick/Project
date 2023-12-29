#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <math.h>

#define EPSILON 1e-6 // Tolerance for comparison

float nearestAdditionTSP(double **distMatrix, int *tour, int startVertex, int N) {
    bool *visited = (bool *)calloc(N, sizeof(bool));
    if (!visited) {
        perror("Memory Allocation Failed for Visited Array");
        exit(EXIT_FAILURE);
    }

    // Initialize the tour with the given start vertex
    tour[0] = startVertex;
    visited[startVertex] = true;

    // The first position after the start vertex is always index 1
    int insertPosition = 1;

    for (int size = 1; size < N; size++) {
        double minDist = INFINITY;
        int nearest = -1;

        #pragma omp parallel shared(visited, tour)
        {
            double localMinDist = INFINITY;
            int localNearest = -1;

            #pragma omp for nowait
            for (int i = 0; i < size; i++) {
                int currentVertex = tour[i];
                for (int j = 0; j < N; j++) {
                    if (!visited[j] && distMatrix[currentVertex][j] < localMinDist) {
                        localMinDist = distMatrix[currentVertex][j];
                        localNearest = j;
                    }
                }
            }

            // Critical section to update global minimum
            #pragma omp critical
            {
                if (localMinDist < minDist) {
                    minDist = localMinDist;
                    nearest = localNearest;
                }
            }
        }
        // Insert the nearest vertex at the position that minimizes the increase in distance between vertex and either side
        double bestDistance = INFINITY;

        for (int pos = 1; pos < size; pos++) { //start from 1 to ensure that the startVertex stays at 0
            double beforeDistance = distMatrix[tour[pos - 1]][nearest] + distMatrix[nearest][tour[pos]];
            double afterDistance  = distMatrix[tour[pos]][nearest] + distMatrix[nearest][tour[pos+1]];
            double minBest = fmin(beforeDistance, afterDistance);
            if (minBest  < bestDistance) {
                bestDistance = minBest;
                insertPosition = (fabs(minBest - beforeDistance)<EPSILON)?pos:pos+1;
            }

        }

        // Insert nearest at the best position found
        for (int i = size; i >= insertPosition; i--) {
            tour[i] = tour[i - 1];
        }
        tour[insertPosition] = nearest;
        visited[nearest] = true;
    }

    // Now that all vertices are visited, close the tour by adding the start vertex at the end if it's a round trip
    tour[N] = startVertex;

    free(visited);


    // Compute the total cost of the tour
    double tourCost = 0.0;
    for (int i = 0; i < N; i++) {
        tourCost += distMatrix[tour[i]][tour[(i + 1)]];
    }

    return tourCost; // Return the total cost of the tour

}