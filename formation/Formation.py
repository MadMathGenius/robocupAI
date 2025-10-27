import numpy as np

def GenerateBasicFormation():
    """
    Returns a simple static formation for 5 players:
    Goalkeeper + 2 Defenders + 2 Attackers
    """
    formation = [
        np.array([-13, 0]),  # Goalkeeper
        np.array([-8, -3]),  # Left Defender
        np.array([-8, 3]),   # Right Defender
        np.array([-3, -2]),  # Left Forward
        np.array([-3, 2])    # Right Forward
    ]
    return formation

def GenerateDynamicFormation(strategyData, offset_x=0.0):
    """
    Generates a dynamic formation based on teammate positions and strategy.
    
    Parameters
    ----------
    strategyData : Strategy
        Object containing game state information (teammate positions, ball, etc.)
    offset_x : float
        Optional x-offset to shift the formation forward/backward
    
    Returns
    -------
    formation_positions : list of np.array
        List of 2D positions for all teammates
    """
    num_players = len(strategyData.teammate_positions)
    formation_positions = []

    # Define a default formation spread
    y_spread = np.linspace(-5, 5, num_players)
    x_base = np.array([-13, -8, -8, -3, -3])[:num_players] + offset_x

    for i in range(num_players):
        formation_positions.append(np.array([x_base[i], y_spread[i]]))

    return formation_positions
