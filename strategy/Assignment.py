import numpy as np

def role_assignment(teammate_positions, formation_positions): 

    # print("\n--- Current team positions ---")
    # for i, pos in enumerate(teammate_positions):
    #     print(f"Player {i} → {pos}")


    # print("\n--- Current formation positions ---")
    # for i, pos in enumerate(formation_positions):
    #     print(f"Role {i} → {pos}")



    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------#

    n = len(teammate_positions)
    assert n == len(formation_positions), "Both input lists must be the same size."

    # Compute player preferences based on Euclidean distance
    players_preferences = {}
    for i in range(n):
        distances = []
        for j in range(n):
            dist = np.linalg.norm(teammate_positions[i] - formation_positions[j])
            distances.append((dist, j))
        # Sort roles by ascending distance (closest first)
        distances.sort(key=lambda x: x[0])
        players_preferences[i] = [role for (_, role) in distances]

    # Compute formation preferences (roles prefer closer players)
    roles_preferences = {}
    for j in range(n):
        distances = []
        for i in range(n):
            dist = np.linalg.norm(teammate_positions[i] - formation_positions[j])
            distances.append((dist, i))
        # Sort players by ascending distance
        distances.sort(key=lambda x: x[0])
        roles_preferences[j] = [player for (_, player) in distances]

    # Initialize all players and roles as free
    free_players = list(range(n))
    current_matches = {j: None for j in range(n)}
    proposals = {i: [] for i in range(n)}

    # Gale–Shapley Matching
    while free_players:
        player = free_players[0]
        prefs = players_preferences[player]

        # Find the first role this player hasn’t proposed to yet
        for role in prefs:
            if role not in proposals[player]:
                proposals[player].append(role)

                # If role is free, match them
                if current_matches[role] is None:
                    current_matches[role] = player
                    free_players.pop(0)
                else:
                    current_player = current_matches[role]
                    # Role prefers the new proposer if they rank higher
                    if roles_preferences[role].index(player) < roles_preferences[role].index(current_player):
                        current_matches[role] = player
                        free_players.pop(0)
                        free_players.append(current_player)
                break

    # Build output dictionary {unum : np.ndarray([x, y])}
    point_preferences = {}
    for role, player in current_matches.items():
        point_preferences[player + 1] = formation_positions[role]

    # print("\n--- Running Stable Marriage Role Assignment ---")
    # for k, v in point_preferences.items():
    #     print(f"Player {k} → {v}")


    return point_preferences


