def render_grid(grid):
    for row in grid:
        print(' '.join(row))
    print()

def clear_console():
    print("\033[H\033[J", end="")

def get_direction_input():
    direction = input("Enter direction (WASD): ").strip().upper()
    if direction in ['W', 'A', 'S', 'D']:
        return direction
    else:
        print("Invalid input. Please enter W, A, S, or D.")
        return get_direction_input()