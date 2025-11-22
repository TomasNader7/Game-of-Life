/* game_of_life_mpi.c
 *
 * Parallel Conway's Game of Life (64x64) using MPI and row-block decomposition.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 64
#define MAX_GENERATIONS 10

/* Count neighbors using local block + halo rows */
int count_neighbors_parallel(int *local_current,
                             int *halo_up,
                             int *halo_down,
                             int rows_per_proc,
                             int start_row,
                             int end_row,
                             int global_row,
                             int local_row,
                             int col)
{
    int count = 0;

    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;

            int nr = global_row + dr;
            int nc = col + dc;

            if (nr < 0 || nr >= N || nc < 0 || nc >= N)
                continue;

            int value;
            if (nr < start_row) {
                value = halo_up[nc];
            } else if (nr > end_row) {
                value = halo_down[nc];
            } else {
                int local_neighbor_row = nr - start_row;
                value = local_current[local_neighbor_row * N + nc];
            }
            count += value;
        }
    }
    return count;
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: N=%d not divisible by p=%d\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    /* Root reads full grid */
    static int global_grid[N][N];

    if (rank == 0) {
        FILE *fin = fopen("input.txt", "r");
        if (!fin) {
            perror("input.txt");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (fscanf(fin, "%d", &global_grid[i][j]) != 1) {
                    fprintf(stderr, "Error reading input.txt at (%d,%d)\n", i, j);
                    fclose(fin);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        fclose(fin);

        /* enforce dead border */
        for (int i = 0; i < N; i++) {
            global_grid[0][i]   = 0;
            global_grid[N-1][i] = 0;
            global_grid[i][0]   = 0;
            global_grid[i][N-1] = 0;
        }
    }

    /* Allocate local storage + halos */
    int *local_current = (int *)malloc(rows_per_proc * N * sizeof(int));
    int *local_next    = (int *)malloc(rows_per_proc * N * sizeof(int));
    int *halo_up       = (int *)malloc(N * sizeof(int));
    int *halo_down     = (int *)malloc(N * sizeof(int));

    if (!local_current || !local_next || !halo_up || !halo_down) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Scatter initial rows from global_grid to each process */
    MPI_Scatter(&global_grid[0][0],
                rows_per_proc * N, MPI_INT,
                local_current,
                rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int global_changed = 1;
    int generations_done = 0;

    for (int gen = 1; gen <= MAX_GENERATIONS; gen++) {

        /* 1) Halo exchange */
        int up_rank   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
        int down_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        MPI_Sendrecv(&local_current[0], N, MPI_INT,
                     up_rank, 0,
                     halo_up, N, MPI_INT,
                     up_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&local_current[(rows_per_proc-1)*N], N, MPI_INT,
                     down_rank, 1,
                     halo_down, N, MPI_INT,
                     down_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank == 0)
            for (int j = 0; j < N; j++) halo_up[j] = 0;
        if (rank == size - 1)
            for (int j = 0; j < N; j++) halo_down[j] = 0;

        /* 2) Compute next generation locally */
        int start_row = rank * rows_per_proc;
        int end_row   = start_row + rows_per_proc - 1;

        int local_changed = 0;

        for (int li = 0; li < rows_per_proc; li++) {
            int global_i = start_row + li;

            for (int j = 0; j < N; j++) {

                /* Global border always dead */
                if (global_i == 0 || global_i == N-1 || j == 0 || j == N-1) {
                    local_next[li*N + j] = 0;
                    continue;
                }

                int neighbors = count_neighbors_parallel(
                    local_current, halo_up, halo_down,
                    rows_per_proc, start_row, end_row,
                    global_i, li, j
                );

                int cell = local_current[li*N + j];
                int new_cell;

                if (cell == 1) {
                    if (neighbors < 2 || neighbors > 3)
                        new_cell = 0;
                    else
                        new_cell = 1;
                } else {
                    new_cell = (neighbors == 3) ? 1 : 0;
                }

                if (new_cell != cell)
                    local_changed = 1;

                local_next[li*N + j] = new_cell;
            }
        }

        /* 3) Check if *any* process changed */
        MPI_Allreduce(&local_changed, &global_changed,
                      1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        generations_done = gen;

        /* 4) Swap buffers */
        int *tmp = local_current;
        local_current = local_next;
        local_next = tmp;

        if (!global_changed)
            break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double parallel_time = t1 - t0;

    if (rank == 0) {
        printf("--- PARALLEL RESULTS (p = %d) ---\n", size);
        printf("Parallel generations executed: %d\n", generations_done);
        printf("Parallel runtime: %f seconds\n", parallel_time);
    }

    free(local_current);
    free(local_next);
    free(halo_up);
    free(halo_down);

    MPI_Finalize();
    return 0;
}
