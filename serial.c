/* game_of_life_serial.c
 *
 * Serial version of Conway's Game of Life for a 64x64 grid.
 */

#include <stdio.h>
#include <stdlib.h>

#define N 64
#define MAX_GENERATIONS 10

/* Count the alive neighbors of cell (row, col) in 'grid'. */
int count_neighbors(int grid[N][N], int row, int col) {
    int count = 0;

    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) {
                continue; // skip the cell itself
            }

            int r = row + dr;
            int c = col + dc;

            // Check boundaries: if outside, just ignore (treated as dead)
            if (r < 0 || r >= N || c < 0 || c >= N) {
                continue;
            }

            count += grid[r][c];
        }
    }

    return count;
}

/* Print the grid to the screen (stdout). */
void print_grid(int grid[N][N], int generation) {
    printf("Generation %d:\n", generation);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Write a generation to a file (append mode). */
void write_grid_to_file(FILE *fout, int grid[N][N], int generation) {
    fprintf(fout, "Generation %d:\n", generation);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fout, "%d ", grid[i][j]);
        }
        fprintf(fout, "\n");
    }
    fprintf(fout, "\n");
}

int main(void) {
    int current[N][N];
    int next[N][N];

    // Open input file and read initial configuration
    FILE *fin = fopen("input.txt", "r");
    if (!fin) {
        perror("Error opening input.txt");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(fin, "%d", &current[i][j]) != 1) {
                fprintf(stderr, "Error reading input.txt at (%d, %d)\n", i, j);
                fclose(fin);
                return 1;
            }
        }
    }
    fclose(fin);

    // Enforce that outer border is always dead
    for (int i = 0; i < N; i++) {
        current[0][i] = 0;
        current[N - 1][i] = 0;
        current[i][0] = 0;
        current[i][N - 1] = 0;
    }

    // Open output file
    FILE *fout = fopen("output.txt", "w");
    if (!fout) {
        perror("Error opening output.txt");
        return 1;
    }

    // Generation 0: print and write initial grid
    print_grid(current, 0);
    write_grid_to_file(fout, current, 0);

    int generation;
    for (generation = 1; generation <= MAX_GENERATIONS; generation++) {
        // Compute next generation
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                // Border cells always dead:
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                    next[i][j] = 0;
                    continue;
                }

                int alive_neighbors = count_neighbors(current, i, j);
                int cell = current[i][j];

                if (cell == 1) {
                    if (alive_neighbors < 2) {
                        next[i][j] = 0; // dies (loneliness)
                    } else if (alive_neighbors == 2 || alive_neighbors == 3) {
                        next[i][j] = 1; // survives
                    } else { // alive_neighbors >= 4
                        next[i][j] = 0; // dies (overcrowding)
                    }
                } else {
                    // cell is dead
                    if (alive_neighbors == 3) {
                        next[i][j] = 1; // birth
                    } else {
                        next[i][j] = 0; // stays dead
                    }
                }
            }
        }

        // Check if there is any change between current and next
        int changed = 0;
        for (int i = 0; i < N && !changed; i++) {
            for (int j = 0; j < N; j++) {
                if (current[i][j] != next[i][j]) {
                    changed = 1;
                    break;
                }
            }
        }

        // Copy next into current
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                current[i][j] = next[i][j];
            }
        }

        // Print and write this generation
        print_grid(current, generation);
        write_grid_to_file(fout, current, generation);

        if (!changed) {
            printf("No change detected at generation %d. Stopping.\n", generation);
            break;
        }
    }

    fclose(fout);
    return 0;
}
