from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation, GenerateDynamicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
         # Decide whether to pass if opponents are close
        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        # Set direction and distance, defaulting to stored values
        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance
        # If abort is requested, stop immediately
        if abort:
            return True

        if self.kick_distance < 1.0:
            return self.behavior.execute("Dribble",None,None)
        
        if self.fat_proxy_cmd is not None:# fat proxy behavior
            return self.fat_proxy_kick()

        else: #normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        
    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



    # def select_skill(self, strategyData):
    #     drawer = self.world.draw
    #     path_draw_options = self.path_manager.draw_options

    #     # ---------------- Constants ----------------
    #     GOAL_BOX_X_MIN = -15
    #     GOAL_BOX_X_MAX = -10
    #     GOAL_BOX_Y_MIN = -5
    #     GOAL_BOX_Y_MAX = 5
    #     GOAL_POS = np.array([15, 0])
    #     STEP_SIZE = 0.5
    #     MIN_DISTANCE = 1.5
    #     CENTRAL_PLAYER_UNUM = 2  # player who stays behind
    #     FRONT_DISTANCE = 1.0     # small forward nudge
    #     SIDE_SPREAD = 2.0        # lateral spread

    #     #------------------------------------------------------
    #     # Role Assignment
    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
    #     else:
    #         drawer.clear("status")

    #     formation_positions = GenerateDynamicFormation(strategyData)
    #     point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)

    #     strategyData.my_desired_position = point_preferences[strategyData.player_unum]
    #     strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
    #         strategyData.my_desired_position
    #     )
    #     drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

    #     # ---------------- Smooth Movement ----------------
    #     if not hasattr(self, "prev_positions"):
    #         self.prev_positions = [pos.copy() for pos in strategyData.teammate_positions]

    #     ball_pos = strategyData.ball_2d
    #     smooth_positions = []

    #     # Determine closest player to ball
    #     distances_to_ball = [np.linalg.norm(pos - ball_pos) for pos in self.prev_positions]
    #     chaser_unum = np.argmin(distances_to_ball) + 1  # unum starts at 1

    #     # ---------------- Update positions ----------------
    #     for i, formation_target in enumerate(point_preferences, start=1):
    #         current_pos = self.prev_positions[i - 1]

    #         # Goalkeeper
    #         if i == 1:
    #             desired_pos = formation_target
    #         # Central player stays behind
    #         elif i == CENTRAL_PLAYER_UNUM:
    #             desired_pos = np.array([0, 0])
    #         # Ball chaser moves directly to ball
    #         elif i == chaser_unum:
    #             desired_pos = ball_pos.copy()
    #         # Supporting players move relative to ball carrier
    #         else:
    #             carrier_pos = self.prev_positions[chaser_unum - 1]
    #             desired_pos = formation_target.copy()

    #             # Prevent moving past the ball carrier
    #             if desired_pos[0] > carrier_pos[0]:
    #                 desired_pos[0] = carrier_pos[0] - 0.5  # slightly behind

    #             # Lateral spread
    #             side_offset = SIDE_SPREAD * ((i % 2) * 2 - 1)
    #             desired_pos[1] = carrier_pos[1] + side_offset

    #         # Smooth movement
    #         move_vector = desired_pos - current_pos
    #         if np.linalg.norm(move_vector) > STEP_SIZE:
    #             move_vector = move_vector / np.linalg.norm(move_vector) * STEP_SIZE
    #         new_pos = current_pos + move_vector

    #         # Goalkeeper box restriction
    #         if i == 1:
    #             new_pos[0] = np.clip(new_pos[0], GOAL_BOX_X_MIN, GOAL_BOX_X_MAX)
    #             new_pos[1] = np.clip(new_pos[1], GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)

    #         smooth_positions.append(new_pos)
    #         self.prev_positions[i - 1] = new_pos

    #     # ---------------- Maintain minimum spacing ----------------
    #     for i in range(len(smooth_positions)):
    #         for j in range(i + 1, len(smooth_positions)):
    #             diff = smooth_positions[i] - smooth_positions[j]
    #             if np.linalg.norm(diff) < MIN_DISTANCE:
    #                 adjust = (diff / np.linalg.norm(diff)) * (MIN_DISTANCE - np.linalg.norm(diff)) / 2
    #                 smooth_positions[i] += adjust
    #                 smooth_positions[j] -= adjust

    #     # ---------------- Helper: Find best forward pass ----------------
    #     def find_best_pass_target(smooth_positions, ball_pos, active_unum):
    #         best_score = -float('inf')
    #         best_target = GOAL_POS
    #         pass_intensity = 1.0

    #         for i, teammate_pos in enumerate(smooth_positions, start=1):
    #             if i == active_unum or i == 1:
    #                 continue
    #             if teammate_pos[0] <= ball_pos[0]:
    #                 continue
    #             distance = np.linalg.norm(teammate_pos - ball_pos)
    #             to_goal = GOAL_POS - teammate_pos
    #             angle_score = np.arctan2(to_goal[1], to_goal[0])
    #             score = 1 / (distance + 0.1) + abs(angle_score)
    #             if score > best_score:
    #                 best_score = score
    #                 best_target = teammate_pos
    #                 pass_intensity = distance
    #         return best_target, pass_intensity

    #     # ---------------- Move / Kick ----------------
    #     smooth_pos = self.prev_positions[strategyData.player_unum - 1]

    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         drawer.annotation((0, 10.5), "Pass Selector Phase", drawer.Color.yellow, "status")

    #         # Ball chaser movement
    #         if strategyData.player_unum == chaser_unum:
    #             dir_to_ball = ball_pos - smooth_pos
    #             if np.linalg.norm(dir_to_ball) > STEP_SIZE:
    #                 dir_to_ball = dir_to_ball / np.linalg.norm(dir_to_ball) * STEP_SIZE
    #             next_pos = smooth_pos + dir_to_ball
    #             self.prev_positions[strategyData.player_unum - 1] = next_pos

    #             # Kick if close to ball
    #             if np.linalg.norm(next_pos - ball_pos) < 0.2:
    #                 target, intensity = find_best_pass_target(smooth_positions, ball_pos, strategyData.player_unum)
    #                 drawer.line(next_pos, target, 2, drawer.Color.red, "pass line")
    #                 orientation_angle = np.arctan2((target - next_pos)[1], (target - next_pos)[0])
    #                 return self.kickTarget(strategyData, ball_pos, target, intensity, orientation=orientation_angle)
    #             else:
    #                 orientation_angle = np.arctan2((GOAL_POS - next_pos)[1], (GOAL_POS - next_pos)[0])
    #                 return self.move(next_pos, orientation=orientation_angle)
    #         else:
    #             # Supporting players move toward formation / relative positions
    #             orientation_angle = np.arctan2((ball_pos - smooth_pos)[1], (ball_pos - smooth_pos)[0])
    #             return self.move(smooth_pos, orientation=orientation_angle)
    #     else:
    #         drawer.clear("pass line")
    #         orientation_angle = np.arctan2((ball_pos - smooth_pos)[1], (ball_pos - smooth_pos)[0])
    #         return self.move(smooth_pos, orientation=orientation_angle)


    def select_skill(self, strategyData):
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options

        # ---------------- Goalkeeper restriction box ----------------
        GOAL_BOX_X_MIN = -15
        GOAL_BOX_X_MAX = -10
        GOAL_BOX_Y_MIN = -5
        GOAL_BOX_Y_MAX = 5
        step_size = 0.5
        min_distance = 1.5
        GOAL_POS = np.array([15, 0])
        CENTRAL_PLAYER_UNUM = 3  # safety/central back player
        SHOOTING_DISTANCE = 1.0   # distance to goal to attempt shot

        # ------------------------------------------------------
        # Role Assignment Annotation
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
        else:
            drawer.clear("status")

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
            strategyData.my_desired_position
        )

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

        # ---------------- Smooth Movement ----------------
        if not hasattr(self, "prev_positions"):
            self.prev_positions = [pos.copy() for pos in strategyData.teammate_positions]

        ball_pos = strategyData.ball_2d.copy()
        ball_x = ball_pos[0]
        smooth_positions = []

        # Determine closest player to ball (current ball carrier)
        distances_to_ball = [np.linalg.norm(pos - ball_pos) for pos in self.prev_positions]
        ball_carrier_unum = np.argmin(distances_to_ball) + 1  # unum starts at 1

        # ---------------- Assign striker roles ----------------
        # Assuming strikers are players 3,4,5 (middle=3, left=4, right=5)
        MIDDLE_STRIKER = 2
        LEFT_STRIKER = 4
        RIGHT_STRIKER = 5

        # Lateral offsets for left/right relative to middle striker
        LEFT_OFFSET = np.array([0, 2.0])
        RIGHT_OFFSET = np.array([0, -2.0])

        # Desired positions for strikers
        desired_positions = {}
        desired_positions[MIDDLE_STRIKER] = GOAL_POS.copy()  # Middle always toward goal
        desired_positions[LEFT_STRIKER] = self.prev_positions[MIDDLE_STRIKER - 1] + LEFT_OFFSET
        desired_positions[RIGHT_STRIKER] = self.prev_positions[MIDDLE_STRIKER - 1] + RIGHT_OFFSET

        # ---------------- Iterate over all players ----------------
        for i, formation_target in enumerate(point_preferences, start=1):
            current_pos = self.prev_positions[i - 1]

            # Determine desired position
            if i == 1:  # goalkeeper
                desired_pos = formation_target
            elif i == CENTRAL_PLAYER_UNUM:
                desired_pos = np.array([0, 0]) if ball_x > 0 else formation_target
            elif i in [MIDDLE_STRIKER, LEFT_STRIKER, RIGHT_STRIKER]:
                desired_pos = desired_positions[i]
            elif i == ball_carrier_unum:
                # Ball carrier's target: check if middle striker is ready for pass
                if ball_carrier_unum != MIDDLE_STRIKER:
                    # Only pass forward if middle striker ahead in x-direction
                    if self.prev_positions[MIDDLE_STRIKER - 1][0] > current_pos[0]:
                        # Minimal forward pass
                        desired_pos = self.prev_positions[MIDDLE_STRIKER - 1]
                    else:
                        # Move slightly forward toward goal
                        desired_pos = current_pos + np.array([0.5, 0.0])
                else:
                    # Middle striker dribbles directly toward goal
                    desired_pos = current_pos + np.array([0.5, 0.0])
            else:
                # Supporting players maintain formation + small offset
                offset = np.array([0, 2 * ((i % 2) * 2 - 1)])
                desired_pos = formation_target + offset

            # Smooth movement
            move_vector = desired_pos - current_pos
            if np.linalg.norm(move_vector) > step_size:
                move_vector = move_vector / np.linalg.norm(move_vector) * step_size
            new_pos = current_pos + move_vector

            # Goalkeeper restriction
            if i == 1:
                new_pos[0] = np.clip(new_pos[0], GOAL_BOX_X_MIN, GOAL_BOX_X_MAX)
                new_pos[1] = np.clip(new_pos[1], GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)

            smooth_positions.append(new_pos)
            self.prev_positions[i - 1] = new_pos

        # Maintain minimum spacing
        for i in range(len(smooth_positions)):
            for j in range(i + 1, len(smooth_positions)):
                diff = smooth_positions[i] - smooth_positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_distance and dist > 0:
                    adjust = (diff / dist) * (min_distance - dist) / 2
                    smooth_positions[i] += adjust
                    smooth_positions[j] -= adjust
                elif dist == 0:
                    nudge = np.random.uniform(-0.1, 0.1, size=2)
                    smooth_positions[i] += nudge
                    smooth_positions[j] -= nudge

        # ---------------- Move Active Player ----------------
        smooth_pos = self.prev_positions[strategyData.player_unum - 1]

        # If I'm the ball carrier
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Attack Phase", drawer.Color.yellow, "status")

            # ---------------- Shooting Logic ----------------
            distance_to_goal = np.linalg.norm(GOAL_POS - smooth_pos)
            if distance_to_goal <= SHOOTING_DISTANCE:
                return self.kickTarget(strategyData, strategyData.mypos, GOAL_POS)

            # ---------------- Passing Logic ----------------
            if strategyData.player_unum in [LEFT_STRIKER, RIGHT_STRIKER]:
                # Always pass to middle striker forward
                target = self.prev_positions[MIDDLE_STRIKER - 1]
                return self.kickTarget(strategyData, strategyData.mypos, target)
            elif strategyData.player_unum == MIDDLE_STRIKER:
                # Middle striker dribbles toward goal
                return self.move(smooth_pos, orientation=strategyData.ball_dir)
            else:
                # Other active player: move to smooth position
                return self.move(smooth_pos, orientation=strategyData.ball_dir)

        else:
            # Non-active players move to smooth positions
            drawer.clear("pass line")
            return self.move(smooth_pos, orientation=strategyData.ball_dir)



    # Had targeting goal working properly
    # def select_skill(self, strategyData):
    #     drawer = self.world.draw
    #     path_draw_options = self.path_manager.draw_options

    #     # ---------------- Goalkeeper restriction box ----------------
    #     GOAL_BOX_X_MIN = -15
    #     GOAL_BOX_X_MAX = -10
    #     GOAL_BOX_Y_MIN = -5
    #     GOAL_BOX_Y_MAX = 5
    #     step_size = 0.5
    #     min_distance = 1.5
    #     GOAL_POS = np.array([15, 0])
    #     CENTRAL_PLAYER_UNUM = 2  # safety/central back player
    #     SHOOTING_DISTANCE = 4.0   # distance to goal to attempt shot

    #     # ------------------------------------------------------
    #     # Role Assignment Annotation
    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
    #     else:
    #         drawer.clear("status")

    #     formation_positions = GenerateDynamicFormation(strategyData)
    #     point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
    #     strategyData.my_desired_position = point_preferences[strategyData.player_unum]
    #     strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
    #         strategyData.my_desired_position
    #     )

    #     drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

    #     # ---------------- Smooth Movement ----------------
    #     if not hasattr(self, "prev_positions"):
    #         self.prev_positions = [pos.copy() for pos in strategyData.teammate_positions]

    #     ball_pos = strategyData.ball_2d.copy()
    #     ball_x = ball_pos[0]
    #     smooth_positions = []

    #     # Determine closest player to ball (current ball carrier)
    #     distances_to_ball = [np.linalg.norm(pos - ball_pos) for pos in self.prev_positions]
    #     ball_carrier_unum = np.argmin(distances_to_ball) + 1  # unum starts at 1

    #     # ---------------- Assign striker roles ----------------
    #     # Assuming strikers are players 3,4,5 (middle=3, left=4, right=5)
    #     MIDDLE_STRIKER = 3
    #     LEFT_STRIKER = 4
    #     RIGHT_STRIKER = 5

    #     # Lateral offsets for left/right relative to middle striker
    #     LEFT_OFFSET = np.array([0, 2.0])
    #     RIGHT_OFFSET = np.array([0, -2.0])

    #     # Desired positions for strikers
    #     desired_positions = {}
    #     desired_positions[MIDDLE_STRIKER] = GOAL_POS.copy()  # Middle always toward goal
    #     desired_positions[LEFT_STRIKER] = self.prev_positions[MIDDLE_STRIKER - 1] + LEFT_OFFSET
    #     desired_positions[RIGHT_STRIKER] = self.prev_positions[MIDDLE_STRIKER - 1] + RIGHT_OFFSET

    #     # ---------------- Iterate over all players ----------------
    #     for i, formation_target in enumerate(point_preferences, start=1):
    #         current_pos = self.prev_positions[i - 1]

    #         # Determine desired position
    #         if i == 1:  # goalkeeper
    #             desired_pos = formation_target
    #         elif i == CENTRAL_PLAYER_UNUM:
    #             desired_pos = np.array([0, 0]) if ball_x > 0 else formation_target
    #         elif i in [MIDDLE_STRIKER, LEFT_STRIKER, RIGHT_STRIKER]:
    #             desired_pos = desired_positions[i]
    #         elif i == ball_carrier_unum:
    #             # Ball carrier's target: check if middle striker is ready for pass
    #             if ball_carrier_unum != MIDDLE_STRIKER:
    #                 # Only pass forward if middle striker ahead in x-direction
    #                 if self.prev_positions[MIDDLE_STRIKER - 1][0] > current_pos[0]:
    #                     # Minimal forward pass
    #                     desired_pos = self.prev_positions[MIDDLE_STRIKER - 1]
    #                 else:
    #                     # Move slightly forward toward goal
    #                     desired_pos = current_pos + np.array([0.5, 0.0])
    #             else:
    #                 # Middle striker dribbles directly toward goal
    #                 desired_pos = current_pos + np.array([0.5, 0.0])
    #         else:
    #             # Supporting players maintain formation + small offset
    #             offset = np.array([0, 2 * ((i % 2) * 2 - 1)])
    #             desired_pos = formation_target + offset

    #         # Smooth movement
    #         move_vector = desired_pos - current_pos
    #         if np.linalg.norm(move_vector) > step_size:
    #             move_vector = move_vector / np.linalg.norm(move_vector) * step_size
    #         new_pos = current_pos + move_vector

    #         # Goalkeeper restriction
    #         if i == 1:
    #             new_pos[0] = np.clip(new_pos[0], GOAL_BOX_X_MIN, GOAL_BOX_X_MAX)
    #             new_pos[1] = np.clip(new_pos[1], GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)

    #         smooth_positions.append(new_pos)
    #         self.prev_positions[i - 1] = new_pos

    #     # Maintain minimum spacing
    #     for i in range(len(smooth_positions)):
    #         for j in range(i + 1, len(smooth_positions)):
    #             diff = smooth_positions[i] - smooth_positions[j]
    #             dist = np.linalg.norm(diff)
    #             if dist < min_distance and dist > 0:
    #                 adjust = (diff / dist) * (min_distance - dist) / 2
    #                 smooth_positions[i] += adjust
    #                 smooth_positions[j] -= adjust
    #             elif dist == 0:
    #                 nudge = np.random.uniform(-0.1, 0.1, size=2)
    #                 smooth_positions[i] += nudge
    #                 smooth_positions[j] -= nudge

    #     # ---------------- Move Active Player ----------------
    #     smooth_pos = self.prev_positions[strategyData.player_unum - 1]

    #     # If I'm the ball carrier
    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         drawer.annotation((0, 10.5), "Attack Phase", drawer.Color.yellow, "status")

    #         # ---------------- Shooting Logic ----------------
    #         distance_to_goal = np.linalg.norm(GOAL_POS - smooth_pos)
    #         if distance_to_goal <= SHOOTING_DISTANCE:
    #             return self.kickTarget(strategyData, strategyData.mypos, GOAL_POS)

    #         # ---------------- Passing Logic ----------------
    #         if strategyData.player_unum in [LEFT_STRIKER, RIGHT_STRIKER]:
    #             # Always pass to middle striker forward
    #             target = self.prev_positions[MIDDLE_STRIKER - 1]
    #             return self.kickTarget(strategyData, strategyData.mypos, target)
    #         elif strategyData.player_unum == MIDDLE_STRIKER:
    #             # Middle striker dribbles toward goal
    #             return self.move(smooth_pos, orientation=strategyData.ball_dir)
    #         else:
    #             # Other active player: move to smooth position
    #             return self.move(smooth_pos, orientation=strategyData.ball_dir)

    #     else:
    #         # Non-active players move to smooth positions
    #         drawer.clear("pass line")
    #         return self.move(smooth_pos, orientation=strategyData.ball_dir)


        

    #Implemented dribbling and shooting logic but had issues with role assignment and smooth movement

    # def select_skill(self, strategyData):
    #     drawer = self.world.draw
    #     path_draw_options = self.path_manager.draw_options

    #     # ---------------- Constants ----------------
    #     GOAL_BOX_X_MIN = -15
    #     GOAL_BOX_X_MAX = -10
    #     GOAL_BOX_Y_MIN = -5
    #     GOAL_BOX_Y_MAX = 5
    #     step_size = 0.5
    #     min_distance = 1.5
    #     GOAL_POS = np.array([15, 0])
    #     CENTRAL_PLAYER_UNUM = 2
    #     ATTACKING_HALF_X = 0.0
    #     DRIBBLE_SPEED = 0.4
    #     BALL_PUSH_DIST = 0.6
    #     SHOOTING_DISTANCE = 3.0

    #     # ---------------- Role Assignment ----------------
    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
    #     else:
    #         drawer.clear("status")

    #     formation_positions = GenerateDynamicFormation(strategyData)
    #     point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
    #     strategyData.my_desired_position = point_preferences[strategyData.player_unum]
    #     strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
    #         strategyData.my_desired_position
    #     )
    #     drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

    #     # ---------------- Initialize tracking ----------------
    #     if not hasattr(self, "prev_positions"):
    #         self.prev_positions = [pos.copy() for pos in strategyData.teammate_positions]
    #     if not hasattr(self, "ball_carrier_unum"):
    #         self.ball_carrier_unum = None

    #     ball_pos = strategyData.ball_2d.copy()
    #     smooth_positions = []

    #     # ---------------- Find closest player to the ball ----------------
    #     distances_to_ball = [np.linalg.norm(pos - ball_pos) for pos in self.prev_positions]
    #     chaser_unum = np.argmin(distances_to_ball) + 1  # unum starts at 1

    #     # ---------------- Maintain ball control ----------------
    #     if self.ball_carrier_unum is None:
    #         self.ball_carrier_unum = chaser_unum
    #     else:
    #         carrier_pos = self.prev_positions[self.ball_carrier_unum - 1]
    #         if np.linalg.norm(carrier_pos - ball_pos) > 2.0:
    #             # Too far, reassign control
    #             self.ball_carrier_unum = chaser_unum

    #     # ---------------- Iterate over players ----------------
    #     for i, formation_target in enumerate(point_preferences, start=1):
    #         current_pos = self.prev_positions[i - 1]

    #         # Goalkeeper stays limited
    #         if i == 1:
    #             desired_pos = formation_target
    #         elif ball_pos[0] > ATTACKING_HALF_X and i == CENTRAL_PLAYER_UNUM:
    #             desired_pos = np.array([0, 0])  # defensive safety
    #         elif i == self.ball_carrier_unum:
    #             # ---------------- Dribbling logic ----------------
    #             direction_to_goal = (GOAL_POS - current_pos)
    #             direction_to_goal /= np.linalg.norm(direction_to_goal)

    #             dist_to_goal = np.linalg.norm(ball_pos - GOAL_POS)
    #             if dist_to_goal > SHOOTING_DISTANCE:
    #                 # Dribble toward goal (ball moves slightly ahead)
    #                 desired_pos = current_pos + direction_to_goal * DRIBBLE_SPEED
    #                 new_ball_pos = current_pos + direction_to_goal * (DRIBBLE_SPEED + BALL_PUSH_DIST)
    #                 strategyData.ball_2d = new_ball_pos
    #                 ball_pos = new_ball_pos  # update for subsequent players
    #                 drawer.line(current_pos, desired_pos, 2, drawer.Color.green, "dribble path")
    #             else:
    #                 # In shooting range
    #                 desired_pos = current_pos
    #                 drawer.annotation((0, 9.5), "Shooting Range!", drawer.Color.red, "status")
    #         else:
    #             # ---------------- Supporting players ----------------
    #             carrier_pos = self.prev_positions[self.ball_carrier_unum - 1]
    #             offset_side = ((i % 2) * 2 - 1) * 3.0  # alternate left/right
    #             offset_forward = 4.0
    #             desired_pos = carrier_pos + np.array([offset_forward, offset_side])

    #         # ---------------- Smooth Movement ----------------
    #         move_vector = desired_pos - current_pos
    #         if np.linalg.norm(move_vector) > step_size:
    #             move_vector = move_vector / np.linalg.norm(move_vector) * step_size
    #         new_pos = current_pos + move_vector

    #         # ---------------- Goalkeeper restriction ----------------
    #         if i == 1:
    #             new_pos[0] = np.clip(new_pos[0], GOAL_BOX_X_MIN, GOAL_BOX_X_MAX)
    #             new_pos[1] = np.clip(new_pos[1], GOAL_BOX_Y_MIN, GOAL_BOX_Y_MAX)

    #         smooth_positions.append(new_pos)
    #         self.prev_positions[i - 1] = new_pos

    #     # ---------------- Maintain spacing between players ----------------
    #     for i in range(len(smooth_positions)):
    #         for j in range(i + 1, len(smooth_positions)):
    #             diff = smooth_positions[i] - smooth_positions[j]
    #             if np.linalg.norm(diff) < min_distance:
    #                 adjust = (diff / np.linalg.norm(diff)) * (min_distance - np.linalg.norm(diff)) / 2
    #                 smooth_positions[i] += adjust
    #                 smooth_positions[j] -= adjust

    #     # ---------------- Active Player Logic ----------------
    #     smooth_pos = self.prev_positions[strategyData.player_unum - 1]

    #     if strategyData.active_player_unum == strategyData.robot_model.unum:
    #         # I'm active player
    #         if strategyData.robot_model.unum == self.ball_carrier_unum:
    #             dist_to_goal = np.linalg.norm(strategyData.ball_2d - GOAL_POS)
    #             if dist_to_goal <= SHOOTING_DISTANCE:
    #                 drawer.annotation((0, 8.5), "Shooting!", drawer.Color.red, "status")
    #                 return self.kickTarget(strategyData, strategyData.mypos, GOAL_POS)
    #             else:
    #                 return self.move(smooth_pos, orientation=strategyData.ball_dir)
    #         else:
    #             return self.move(smooth_pos, orientation=strategyData.ball_dir)
    #     else:
    #         # Other players
    #         return self.move(smooth_pos, orientation=strategyData.ball_dir)






    
        #------------------------------------------------------
        # Example Behaviour
        # target = (15,0) # Opponents Goal

        # if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
        #     drawer.annotation((0,10.5), "Pass Selector Phase" , drawer.Color.yellow, "status")
        # else:
        #     drawer.clear_player()

        # if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
        #     pass_reciever_unum = strategyData.player_unum + 1 # This starts indexing at 1, therefore player 1 wants to pass to player 2
        #     if pass_reciever_unum != 6:
        #         target = strategyData.teammate_positions[pass_reciever_unum-1] # This is 0 indexed so we actually need to minus 1 
        #     else:
        #         target = (15,0) 

        #     drawer.line(strategyData.mypos, target, 2,drawer.Color.red,"pass line")
        #     return self.kickTarget(strategyData,strategyData.mypos,target)
        # else:
        #     drawer.clear("pass line")
        #     return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        































    

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")