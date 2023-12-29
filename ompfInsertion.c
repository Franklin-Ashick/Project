#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <omp.h>

// Farthest Insertion TSP Algorithm with OpenMP
double farthestInsertionTSP(double **distMatrix, int *tour, int startVertex, int N) {
    bool *visited = (bool *)calloc(N, sizeof(bool));
    int tourSize = 1;
    tour[0] = startVertex;
    visited[startVertex] = true;

    // Main loop to insert all vertices into the tour
    while (tourSize < N) {
        double maxDistance = -1.0;
        int maxVertex = -1;
        // Declare structure to hold local results
        struct {
            double maxDistance;
            int maxVertex;

        } local;

        #pragma omp parallel private(local)
        {
            local.maxDistance = -1.0;
            local.maxVertex = -1;
            // Parallel for loop to find the farthest unvisited vertex from the tour
            #pragma omp for nowait
            for (int i = 0; i < N; i++) {
                if (!visited[i]) {
                    for (int j = 0; j < tourSize; j++) {
                        if (distMatrix[tour[j]][i] > local.maxDistance) {
                            local.maxDistance = distMatrix[tour[j]][i];
                            local.maxVertex = i;
                        }
                    }
                }
            }

            // Critical section to update the global maximum
            #pragma omp critical
            {
                if (local.maxDistance > maxDistance) {
                    maxDistance = local.maxDistance;
                    maxVertex = local.maxVertex;
                }
            }
        } // End of parallel region

        double minCost = DBL_MAX;
        int minPos = -1;
        for (int i = 0; i < tourSize; i++) {
            int next = (i + 1) % tourSize;
            double cost = distMatrix[tour[i]][maxVertex] + distMatrix[maxVertex][tour[next]] - distMatrix[tour[i]][tour[next]];    // Calculate the insertion cost

            if (cost < minCost) {
                minCost = cost;
                minPos = i + 1;
            }
        }
        // Insert the vertex into the tour at the specified position
        for (int i = tourSize; i > minPos; i--) {
            tour[i] = tour[i - 1];
        }
        tour[minPos] = maxVertex;

        tourSize++;
        visited[maxVertex] = true;
    }

    free(visited);
    // Add the logic to close the tour by returning to the start vertex
    tour[N] = tour[0];



    // Compute the total cost of the tour
    double tourCost = 0.0;
    for (int i = 0; i < N; i++) {
        tourCost += distMatrix[tour[i]][tour[(i + 1) % (N + 1)]];
    }

    return tourCost; // Return the total cost of the tour
}