# -*- coding: utf-8 -*-
#pdoc --html MothAgent --output-dir ./docs
"""
Created on Fri Apr 5 14:39:22 2024
@author: Thomas

This module defines the MothAgent class which handles autonomous movement and luminosity detection 
in a grid-based environment. The agent is designed to navigate and find bright areas efficiently.

This module is designed to operate on a Sphero BOLT using the spheroV2 SDK Module - passed as Droid.
"""

from spherov2.types import Color
import time
import math

class MothAgent:
    """
    Represents an autonomous agent that navigates a grid-based environment to find areas with high luminosity.

    Attributes:
        droid: A robotic agent instance - sphero BOLT from the spherov2 module.
        bounds (tuple): Centimetres - Grid boundaries as (width, height), virtual walls of the search area.
        grid_size (int): Centimetres -Size of each grid cell.
        environment (dict): Representation of the entire environment, 
        environment[key] is a tuple representing (x,y) locations of the grid.
        environment[val] is a tuple represetning the state of that location
        (visited,lum) where visitied is 0 if not visited or 1 if visited, and lum is the recorded brightness.
        frontier (list): The search frontier used by the searching algorithms.
        done (bool): Indicates terminal state, defaults to False, is set to True when terminal state reached.
        lum (float): Luminosity reading in the current state.
        max_lum (float): Highest luminosity encountered.
        brightest_loc (tuple): Coordinates of the brightest cell found.
        log (file): Data logging file name.
    """
    
    def __init__(self, droid, bounds, grid_size, log_file):
        """
        Initialises the agent with the specified robotic interface, environment boundaries, grid size, and log file.
        Takes care of constrruction the environment dictionary, and setting the current state variables at the 0th timestep.
        Opens the logging file and writes the column headers.
        """
        self.bounds = bounds
        self.grid_size = grid_size
        self.droid = droid
        
        self.x = int(self.droid.get_location()['x'] / self.grid_size)
        self.y = int(self.droid.get_location()['y'] / self.grid_size)
        self.environment = {(x, y): (0, 0) for x in range(bounds[0] // grid_size) for y in range(bounds[1] // grid_size)}

        self.frontier = [(self.x, self.y)]
        
        self.done = False
        
        self.lum = 0
        self.max_lum = 0
        self.brightest_loc = (self.x, self.y)
        
        self.time_step = 0
        self.log_file = log_file
        with open(self.log_file, 'w') as file:
            file.write("time, timestep, x, y, lum\n")
        
    def transition_function(self, action, timeout=15):
        """
        Executes the agenttransition function actions with a timeout mechanism
        to prevent operations from hanging indefinitely.
        
        Parameters
        ----------
        action: tuple
            The action ((x,y)location) - parameter to be passed to the move_to_point method.
        timeout: int 
            Default = 15. The maximum time (in seconds) allowed for for transition before timing out.
        
        Raises
        ------
        Exception: If the transition function failed to move or update state due to environmental factors.
        
        Returns
        -------
        None. Calls functions to directly set state variables and move the droid.
        """
        self.time_step += 1
        
        if not self.move_to_point(action, timeout):
            raise Exception("Error in transition function - unable to move to next state.")
        if not self.update_agent_state():
            raise Exception("Error in transition function - unable to update state variable.")
        self.data_log()
        self.print_grid()

    def update_agent_state(self):
        """ 
        Part of the transition function of the agent. 
        Directly updates the state variables: x, y, lum, environment, max_lum and brightest_loc.
        
        Returns
        -------
        Bool: 
            True if state was update successfully else false.
        """
        new_lum = self.get_lum()
        if new_lum == None: #For error handling in the transition function
            return False
        self.lum = new_lum
        if self.lum > (self.max_lum or 0): #Maintainn track of the brightest location and lum value
            self.max_lum = self.lum
            self.brightest_loc = (self.x, self.y)
            
        self.x = int(self.droid.get_location()['x'] / self.grid_size)
        self.y = int(self.droid.get_location()['y'] / self.grid_size)
        self.environment[(self.x, self.y)] = (1, self.lum)
        
        return True
    
    def get_lum(self, retry_count=0):
        """
        Forms part of the transition function.
        Measures and returns the average luminosity from multiple readings.
        Handles potentially invalid readings through recursive calls.
        
        Parameters
        ----------
        retry_count: int
            Maximum allowable recursion depth when handling bad data.
        
        Returns
        -------
        lum : int
            Integer value of the lum reading, or None in case of bad data
        """
        if retry_count >5:
            return None
        
        lum_readings = []
        for _ in range(10):
            lum_readings.append(self.droid.get_luminosity()['ambient_light'])
            #self.droid.spin(36, 0.1)
            time.sleep(0.01)
        avg_lum = sum(lum_readings) // len(lum_readings)
        
        #Sphero BOLT likes to retun nonesense data, recursively call this function in case of nonsense.
        if avg_lum < 0:
            time.sleep(0.5)
            avg_lum = self.get_lum(retry_count+1)
        return avg_lum
        
    def move_to_point(self, waypoint, max_time=30):
        """
        Code in this function adapted from supplied waypoint follower code provided in ZEIT4161 RAS Moodle Page.
        
        Part of the transition function of the agent.
        Moves the agent to the specified grid point.
        Updates the agents state variables upon reaching the new state.
        Gracefully handles faulty data fed from the spherov2 SDK and Sphero BOLT robot acting as droid.
        Will return false and "give up" after max_time seconds.
        
        Parameters
        ----------
        max_time: int (seconds)
            Defaults to 15 - time after whcih the function will break and return false.
        
        Returns
        -------
        Boolean. True if sucesfully reached the location, otherwise returns false to indicate a failure.
        """
        target_x = (waypoint[0] * self.grid_size) + (self.grid_size * 0.5)
        target_y = (waypoint[1] * self.grid_size) + (self.grid_size * 0.5)
        
        #Retry counters for handling bad data from sphero
        retry_count = 0
        max_retries = 5
        
        #timers for preventing infinite hangups if the sphero gets stuck
        start = time.time()
        elapsed=time.time()
        
        while elapsed < start + max_time: 
            elapsed=time.time()
            location = self.droid.get_location()
            x = location['x']
            y = location['y']
            if math.isnan(x) or math.isnan(y):
                if retry_count < max_retries:
                    time.sleep(0.5)
                    retry_count += 1
                    continue
                else:
                    print("Failed to get valid location data after several retries.")
                    return False

            dist_error = math.sqrt((x - target_x)**2 + (y - target_y)**2)
            theta_error = 180 * math.atan2(target_x - x, target_y - y) / math.pi
            self.droid.set_heading(int(theta_error))
    
            try:
                if dist_error > 10:
                    self.droid.set_speed(45)
                elif dist_error < 5:
                    self.droid.set_speed(0)
                    return True
                else:
                    self.droid.set_speed(int(dist_error) + 10)
                time.sleep(0.01)
            except ValueError as e:
                print(f"Error in setting heading: {e}")
                return False
        return False


    def get_valid_actions(self, x, y):
        """
        A utility to the searching algorithms.
        Generates a set of possible allowable actions for navigation of the grid
        environemnt from a location (x,y).
        For the use of this function, re-visiting already visited locations is
        considred to be invalid and not allowable.
        
        Parameters
        ----------
        x: int
            X co-ordinate of the queried location
        y: int
            Y co-ordinate of the queried location
        
        Returns
        -------
        valid_actions: array
            An array containing all valid actions from the queried location given the current environment state.
        """
        valid_actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) in self.environment and self.environment[(new_x, new_y)][0] == 0:
                    valid_actions.append((new_x, new_y))
        return valid_actions

    def brightest_adjacent_cell(self, cell):
        """
        A utility to the searching algorithms.
        Identifies the brightest among the adjacent cells of a given cell.
        If the queried cell is brighter than all it's adjacent cells then 
        the queried cell location will be returned.
        
        Parameters
        ----------
        cell : Tuple
            (x,y) co ordinates of a queried location

        Returns
        -------
        brightest : Tuple.
            (x,y) co-ordinates of the brightest cell adjacent to the queried cell.
        """
        brightest = (cell[0],cell[1])
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = cell[0] + dx, cell[1] + dy
                if (new_x, new_y) in self.environment:
                    if self.environment[(new_x, new_y)][1] > self.environment[brightest][1]:
                        brightest = (new_x, new_y)
        return brightest

    def depth_first_search(self, threshold):
        """
        Perform a depth-first search of the environment.
        Frontier is used as a stack, or single ended queue in this instance.
        Algorithm takes the following steps:
            -Find valid actions at the current state.
            -Push these actions to the frontier.
            -Pop the top action from the frontier.
            -Call the transition function (move_to_point()) with the new action.
            -Cointinue until a specified threshold value is discovered.
        
        The way that valid actions are returned and pushed to the stack results in
        the agent preffering to travel diagonally to extreme points of the environment first.
        
        The search continues until one of the two conditions are met: 
            - A location with luminosity greater than the specified threshold has been found.
            - The entire environment has been searched.
            
        Parameters
        ----------
        threshold: int
            A threshold luminosity reading value to triger breaking out of the search.
            
        Returns
        -------
        None. Calls the transition function (move_to_point()) from within.
        """
        self.set_lights('searching')
        
        self.frontier = []
        while self.max_lum < threshold:
                
            print("getting valid_actions")
            valid_actions = self.get_valid_actions(self.x, self.y)
            print(valid_actions)
            #input("Agent Paused, Press Enter to Carry on...")
            if valid_actions:
                print("Valid Actions Found, Updating Frontier")
                self.frontier.extend([action for action in valid_actions if action not in self.frontier])
            if self.frontier:
                print(f"New Frontier: \n{self.frontier}")
                new_node = self.frontier.pop()
                self.transition_function(new_node)
            else:
                break

    def optimised_greedy_search(self):
        """
        Perform a Greedy search exploration of the environemnt.
        At each step the agent moves in the direction that has the brightest location.
        Algorithm takes the following steps:
            -Given current search location (x,y) determine the valid actions.
            -Take each valid action and update the state variabls for each new state,action.
            -Once all adjacent cells have been visited, determine which adjacent cell to the search location is the brightest.
            -Travel to the brightest location and repreat.
            -If the curent search location is the brightest the terminal state has been reached
            -Set self.done = True
        
        This approach is optimised in the sense that it uses the known luminosity data of previously 
        visited cells to make informed decisions quickly, thus minimizing unnecessary exploration.

        Returns:
            None. Calls the transition function (move_to_point()) from within. Will directly set self.done to True.
        """
        self.set_lights('localising')
        
        maximum = (self.x, self.y)
        
        while not self.done:
            print("getting valid_actions")
            valid_actions = self.get_valid_actions(self.x, self.y)
            print(valid_actions)
            #input("Agent Will Visit adjecent Cells, Press Enter to Carry on...")
            for action in valid_actions:
                self.transition_function(action)
            #input("Agent Will now compute the brightest adjacent cell and drive there, Press Enter to Carry on...")
            brightest = self.brightest_adjacent_cell(maximum)
            self.transition_function(brightest)
            if maximum == brightest:
                print(f"Lamp found at {brightest} with luminosity {self.environment[brightest][1]}")
                self.done = True
            maximum = brightest

    def sweep_search(self):
        """
        This is one of the top level callable search algorithms that the agent may employ.Executes a sweeping coverage of the environment in a snake-like, zigzag pattern to ensure all areas are visited.
        This method will set the self.done variable to True upon reaching the terminal state.
   
        Returns:
        None. The agent's state and the map of the environment are updated internally to reflect the progress of the search.
        """
        self.update_agent_state()
        self.set_lights('coverage')
        zigzag = True
        while any(value[0] == 0 for value in self.environment.values()):    
            if zigzag:
                if self.y < (self.bounds[1] / self.grid_size) - 1:
                    new_target = (self.x, self.y + 1)
                    print("moving North")
                elif self.x < (self.bounds[0] / self.grid_size) - 1:
                    new_target = (self.x + 1, self.y)
                    zigzag = False
                else:
                    print("Edge of the Universe.")
            else:
                if self.y > 0:
                    new_target = (self.x, self.y - 1)
                    print("Moving South")
                elif self.x < (self.bounds[0] / self.grid_size) - 1:
                    new_target = (self.x + 1, self.y)
                    zigzag = True
                else:
                    print("Edge of the Universe.")
                    #Collect any cells we missed along the way
                    for location, status in self.environment.items():
                        if status[0] == 0:  # Checking if the 'visited' status is 0
                            self.transition_function(location)  # Call the transition function with the location

            self.transition_function(new_target)
        
        max_lum_location = max(self.environment, key=lambda k: self.environment[k][1])
        self.transition_function(max_lum_location)
        self.done = True

    def active_sensing(self, threshold = 100):
        """
        Conducts an active sensing search of the environemnt.
        First conduction DFS until a threshhold luminosity is discovered.
        Then implements a greedy search to reach terminal state.
        Find the area of highest luminosity.
        
        Parameters
        ----------
        threshold : int
            
        
        """
        self.update_agent_state()
        self.depth_first_search(threshold)
        print(f"Greater than {threshold} found, switching to active sensing")
        self.optimised_greedy_search()    

    def data_log(self):
        """
        Logs the agent's position, luminosity, and time step to a file.
        """
        
        data = f"{time.time_ns()}, {self.time_step}, {self.x}, {self.y}, {self.lum}\n"
        with open(self.log_file, 'a') as file:
            file.write(data)

    def set_lights(self, style):
        """
        A basic utility function to set the lights.
        Useful for instantaneous visul feedback of the agent's actions to an observer.
        
        Returns
        -------
        None. Directly sets the LED colours through the spheroV2 SDK.
        """
        colours = {
            'coverage': ((75, 0, 130),(0,0,0)),  # Purple
            'searching': ((255, 140, 0),(0,0,0)), # Orange
            'localising': ((0, 150, 0),(75,75,75)),  # Green
            'party1': ((255, 0, 0),(0,0,255)),      # Red
            'party2': ((0, 0, 255),(255,0,0)),       # Blue
            'home':((255,255,255),(255,255,255))  #Bright White
        }
        main_colour, small_colour = colours.get(style, ((0, 0, 0), (0, 0, 0)))

        self.droid.set_main_led(Color(main_colour[0], main_colour[1], main_colour[2]))
        self.droid.set_front_led(Color(small_colour[0], small_colour[1], small_colour[2]))
        self.droid.set_back_led(Color(small_colour[0], small_colour[1], small_colour[2]))

    def print_grid(self):
        """
        Prints a visual representation of the grid environment, with blank squares for un-visited sites,
        and the measured luminosity for visited locations.
        Prints to the terminal using the maintained state variables
        
        returns
        -------
        none. Prints to the terminal.
        """
        num_cells_x = self.bounds[0] // self.grid_size
        num_cells_y = self.bounds[1] // self.grid_size
    
        top_row = ' ' * 5 + ' '.join(f'{x:3d} ' for x in range(num_cells_x))
        print(top_row)
        horizontal_line = ' ' * 4 + '+' + ('----+' * num_cells_x)
        
        for y in reversed(range(num_cells_y)):
            print(horizontal_line)
            row = f'{y:3d} |'
            for x in range(num_cells_x):
                cell = (x, y)
                if self.environment.get(cell, (0, 0))[0] == 1:
                    brightness = int(self.environment[cell][1])
                    row += f'{brightness:3d} |'
                else:
                    row += '    |'
            print(row)
        print(horizontal_line)
