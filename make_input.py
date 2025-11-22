import random

N = 64
density = 0.25  # 25% alive cells

# Create 64x64 grid initialized to 0
grid = [[0 for _ in range(N)] for _ in range(N)]

# Fill inner cells (1..62) with random 0/1 with ~25% 1s
for i in range(1, N - 1):
    for j in range(1, N - 1):
        if random.random() < density:
            grid[i][j] = 1

# Add a glider at position (row, col) = (10, 10)
# Pattern:
# . 1 .
# . . 1
# 1 1 1
r, c = 10, 10
grid[r][c+1]   = 1
grid[r+1][c+2] = 1
grid[r+2][c]   = 1
grid[r+2][c+1] = 1
grid[r+2][c+2] = 1

# Make sure outer border is all zeros (just to be explicit)
for i in range(N):
    grid[0][i]     = 0
    grid[N-1][i]   = 0
    grid[i][0]     = 0
    grid[i][N-1]   = 0

# Write to input.txt
with open("input.txt", "w") as f:
    for i in range(N):
        row_values = " ".join(str(grid[i][j]) for j in range(N))
        f.write(row_values + "\n")

print("input.txt generated with 25% density and a glider.")
