import numpy as np
from strategy.Assignment import role_assignment

# Example teammate and formation positions (like on a small soccer field)
teammate_positions = [
    np.array([1, 2]),
    np.array([0, 1]),
    np.array([6, 1]),
    np.array([2, 5]),
    np.array([5, 4])
]

formation_positions = [
    np.array([0, 0]),
    np.array([2, 2]),
    np.array([4, 2]),
    np.array([6, 3]),
    np.array([3, 5])
]

# Call your stable marriage assignment
results = role_assignment(teammate_positions, formation_positions)

print("\nFINAL MATCH RESULTS:")
for player, position in results.items():
    print(f"Player {player} assigned to {position}")
