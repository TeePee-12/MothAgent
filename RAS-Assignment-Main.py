# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 14:39:22 2024
@author: Thomas

This is the main access point to the MothAgent usage.
It is designed to be run in a standalone terminal window.

This module is designed to operate on a Sphero BOLT using the spheroV2 SDK Module.
"""
import time 
import sys
import traceback
import pandas as pd
import seaborn as sns
from spherov2 import scanner
import matplotlib.pyplot as plt
from MothAgent import MothAgent
from spherov2.toy.bolt import BOLT
from spherov2.sphero_edu import EventType, SpheroEduAPI

def main():
    
    log_file = "Moth_testing.csv"
    
    boundary = input("Define the environment. Set boundary wall length [CM]:\n")
    boundary = [int(boundary),int(boundary)]
    
    grid_size = input("Set Environment Resolution [CM]. Reccomend no less than 25cm:\n")
    grid_size = int(grid_size)
    
    agent_type = input("Agent Type:?\n[C] Coverage   [A] Active Sensing\n")
    if agent_type == "C":
        print("Full Coverage Sweep Search Moth Will Find The Lamp\n\n")
    elif agent_type == "A":
        print("Active Sensing Moth Will Find The Lamp\n\n")
    else:
        print("Invalid Choice, Moth Will Do Nothing\n\n")
    
    toy = scanner.find_toys(toy_names=['SB-B58B'])
    time.sleep(2)
    with SpheroEduAPI(toy[0]) as droid:
        time.sleep(5)
        global log
        
        try:
            moth = MothAgent(droid, boundary, grid_size, log_file)
            while not moth.done:
                
                if agent_type == "C":
                    moth.sweep_search()
                if agent_type == "A":
                    moth.active_sensing()
                heatmap(moth.environment)
            for i in range(10):
                moth.set_lights('party1')
                droid.spin(30, 0.001)
                moth.set_lights('party2')
                droid.spin(-45, 0.001)
            moth.set_lights('home')
            moth.move_to_point((0,0))
            moth.droid.set_heading(0)   
            input("Press Enter to close...")
            print("Moth Agent Terminated.")
                
                
        except KeyboardInterrupt:
            print('E.T. PHONE HOME')
            moth.set_lights('off')
            moth.move_to_point((0,0))
            moth.droid.set_heading(0)   
            input("Press Enter to close...")
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("\nStack trace:")
            traceback.print_tb(exc_traceback)
            print("Exception value:", exc_value)
            print("\nSomething has gone terribly wrong!\nMoth will attempt to return home now.")
            moth.set_lights('off')
            moth.move_to_point((0,0))
            moth.droid.set_heading(0)
            heatmap(moth.environment)
            input("Press Enter to close...")

def heatmap(environment):
    """
    Plots a 3D heatmap of luminosity data. Because printing to a terminal window wasn't good enough.
    """
    data = {'x': [], 'y': [], 'luminosity': []}
    for (x, y), (_, lum) in environment.items():
        data['x'].append(x)
        data['y'].append(y)
        data['luminosity'].append(lum)
    df = pd.DataFrame(data)
 
    lum_matrix = df.pivot_table(index='y', columns='x', values='luminosity', fill_value=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(lum_matrix[::-1], cmap='inferno', vmin=0, annot=False, fmt=".1f")  
    max_lum_location = df.loc[df['luminosity'].idxmax()]
    max_x, max_y = max_lum_location['x'], len(lum_matrix) - max_lum_location['y'] - 1  # Adjusting y for reversed heatmap
    plt.scatter(max_x + 0.5, max_y + 0.5, color='red', marker='x', s=200, label='Highest Luminosity')
    plt.legend()
    plt.title('Luminosity Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show(block=False)
    plt.pause(0.1)

            
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Main Interrupted')


