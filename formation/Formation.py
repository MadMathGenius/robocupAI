import numpy as np

def GenerateBasicFormation():


    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-9, -4]),  # Left Defender
        np.array([-9, 4]),   # Right Defender
        np.array([-1, -3 ]),    # Forward Left
        np.array([-1, 3])      # Forward Right
    ]



    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]

    return formation

def GenerateDynamicFormation(strategyData):
    ball_y, ball_x = strategyData.ball_2d
    side = strategyData.side  # 0 for left, 1 for right

    formation = []

    if ball_x < -5:  # Defensive half
        formation = [
            np.array([-13, 0]),   # Goalkeeper
            np.array([-8, -2]),   # Left defender
            np.array([-7, 3]),    # Right defender
            np.array([-3, 2]),    # Defensive mid
            np.array([-2, -1])    # Support
        ]
    elif -5 <= ball_x <= 5:  # Midfield control
        formation = [
            np.array([-13, 0]),   # Goalkeeper
            np.array([-4, -2]),   # Left mid
            np.array([-3, 3]),    # Right mid
            np.array([3, 7]),     # Attacker left
            np.array([4, -7])     # Attacker right
        ]
    else:  # Attacking third
        formation = [
            np.array([-13, 0]),   # Goalkeeper
            np.array([-2, -3]),   # Support left
            np.array([12, 0]),    # Support right
            np.array([9, 7]),    # Forward left
            np.array([9, -7])    # Forward right
        ]

    # Flip formation if your team is on the right
    if side == 1:
        formation = [np.array([-p[0], p[1]]) for p in formation]

    return formation

