from math import ceil
from opensbt import config
from opensbt.simulation.simulator import DrivingSimulationOutput
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.colors import colorConverter
import numpy as np
from opensbt.visualization.configuration import *


def plot_scenario_gif(parameter_values, 
                      simout: DrivingSimulationOutput, 
                      savePath=None, 
                      fileName=None, 
                      trace_interval=config.DEFAULT_TRACE_INTERVAL):
    
    """This functions visualizes the executed scenario.
    """
    
    if "car_length" in simout.otherParams:
        car_length = float(simout.otherParams["car_length"])
    else:
        car_length = float(config.DEFAULT_CAR_LENGTH)

    if "car_width" in simout.otherParams:
        car_width = float(simout.otherParams["car_width"])
    else:
        car_width = float(config.DEFAULT_CAR_WIDTH)

    if "pedestrian_size" in simout.otherParams:
        pedestrian_size = float(simout.otherParams["pedestrian_size"])
    else:
        pedestrian_size = config.DEFAULT_PEDESTRIAN_SIZE

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    colors = { "ego" : colorConverter.to_rgba('yellow', alpha=0.6),
            "adversary" : colorConverter.to_rgba('red', alpha=0.2),
               "vehicles": colorConverter.to_rgba('green', alpha=0.6),
               "pedestrians": colorConverter.to_rgba('purple', alpha=0.2),
               "traces": colorConverter.to_rgba('black', alpha=0.2)}


    ''' actors =
        { 
            "ego" : <ego_name>,   //ego vehicle
            "adversary": <adv_name>  //primary adversary actor (TODO: remove)
            "vehicles" : <list of actor name that are vehicles> 
            "pedestrians" : <list of actors names that are pedstrians>
        }
     '''
    
    # determine trace_interval
    if trace_interval is not None:
        dif = np.asarray([(simout.times[i+1] - simout.times[i]) for i in range(0,len(simout.times)- 1)])
        avg_dif = np.average(dif)
        skip = ceil(trace_interval/avg_dif)
    else:
        skip = 1
 
    actors = simout.actors
    ego_name = actors["ego"]
    
    if "adversary" in actors:
        adversary_name = actors["adversary"]
    else:
        adversary_name = None

    vehicles_names = actors["vehicles"]
    
    if "pedestrians" in actors:    
        pedestrians_names = actors["pedestrians"]
    else:
        pedestrians_names = None

    actors_names = [ego_name]  + \
                    [adversary_name] if adversary_name is not None else [] + \
                    vehicles_names if vehicles_names is not None else [] + \
                    pedestrians_names if pedestrians_names is not None else []

    "Traces and yaws for actors"
    trace_ego = np.array(simout.location[ego_name][0::skip])  # time series of Ego position
    yaw_ego = np.array(simout.yaw[ego_name][0::skip])  # time series of Ego velocity

    if adversary_name is not None:
        trace_adversary = np.array(simout.location[adversary_name][0::skip])  # time series of adversary position
        yaw_adversary = np.array(simout.yaw[adversary_name][0::skip])
    
    traces_vehicles = [np.array(simout.location[vehicle_name][0::skip]) for vehicle_name in vehicles_names]
    yaws_vehicles = [np.array(simout.yaw[vehicle_name][0::skip]) for vehicle_name in vehicles_names]

    if "pedestrians" in actors:
        traces_pedestrians = [np.array(simout.location[pedestrian_name][0::skip]) for pedestrian_name in pedestrians_names]

    "Cartesian coordinates for actors"
    x_ego = trace_ego[:, 0]
    y_ego = trace_ego[:, 1]
   
    if adversary_name is not None:
        x_adversary = trace_adversary[:, 0]
        y_adversary = trace_adversary[:, 1]

    x_vehicles = [traces_vehicles[i][:, 0] for i in range(len(vehicles_names))]
    y_vehicles = [traces_vehicles[i][:, 1] for i in range(len(vehicles_names))]
   
    if "pedestrians" in actors:
        x_pedestrians = [traces_pedestrians[i][:, 0] for i in range(len(pedestrians_names))]
        y_pedestrians = [traces_pedestrians[i][:, 1] for i in range(len(pedestrians_names))]

    "Lables and titles"
    label_parameters = str(["{:.3f}".format(v) for v in parameter_values])
    plt.title(f"Simulation of scenario {label_parameters}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    '''Finding suitable axis limits for visualization'''
    borders_x = []
    borders_y = []
    for actor_name in actors_names:
        trace_actor = np.array(simout.location[actor_name][0::skip])
        x_actor = trace_actor[:,0]
        y_actor = trace_actor[:,1]
        borders_x.extend([min(x_actor), max(x_actor)])
        borders_y.extend([min(y_actor), max(y_actor)])

    left_border_x = min(borders_x)
    right_border_x = max(borders_x)
    center_x = (left_border_x + right_border_x) / 2

    left_border_y = min(borders_y)
    right_border_y = max(borders_y)
    center_y = (left_border_y + right_border_y) / 2

    maximum_size_of_axis = max((right_border_y - left_border_y), (right_border_x - left_border_x))

    ax.set(xlim=(center_x - maximum_size_of_axis * 0.55, center_x + maximum_size_of_axis * 0.55),
           ylim=(center_y - maximum_size_of_axis * 0.55, center_y + maximum_size_of_axis * 0.55))

    "Rectangle objects for ego and vehicles"
    patch_ego = Rectangle((0, 0), width=car_width, height=car_length, color=colors["ego"])
    patch_ego.set_width(car_width)
    patch_ego.set_height(car_length)

    patches_vehicles = [Rectangle((0, 0), width=car_width, height=car_length, color=colors["vehicles"]) for _ in
                        range(len(vehicles_names))]
    for patch_vehicle in patches_vehicles:
        patch_vehicle.set_width(car_width)
        patch_vehicle.set_height(car_length)
        # TODO: introduce separate dimensions of vehicles

    ''' if adversary car:
    patch_adversary = Rectangle((0, 0), width=car_width, height=car_length, color=colors["adversary"])
    patch_adversary.set_width(car_width)
    patch_adversary.set_height(car_length)
    '''

    "Circle objects for adversary and pedestrians"
    if adversary_name is not None:
        circle_adversary = Circle((0, 0), radius=pedestrian_size, color=colors["adversary"])
        plt.plot(x_adversary, y_adversary, color=colors["traces"])

    if pedestrians_names is not None:
        circles_pedestrians = [Circle((0, 0), radius=pedestrian_size, color=colors["pedestrians"]) for _ in range(len(pedestrians_names))]

    "Plot traces of all objects"
    plt.plot(x_ego, y_ego, color=colors["traces"])

    for key, pedestrian in enumerate(pedestrians_names):
        x_pedestrian = x_pedestrians[key]
        y_pedestrian = y_pedestrians[key]
        plt.plot(x_pedestrian, y_pedestrian, color=colors["traces"])

    for key, vehicle in enumerate(vehicles_names):
        x_vehicle = x_vehicles[key]
        y_vehicle = y_vehicles[key]
        plt.plot(x_vehicle, y_vehicle, color=colors["traces"])

    def update(i):
        # updating ego
        rotation_angle_ego = (yaw_ego[i] - 90) % 360
        patch_ego.set_angle(rotation_angle_ego)
        shift_angle_ego = rotation_angle_ego - 180 / np.pi * np.arctan(car_width / car_length)
        shift_x_ego = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(shift_angle_ego * np.pi / 180)
        shift_y_ego = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(shift_angle_ego * np.pi / 180)
        patch_ego.set_xy([x_ego[i] + shift_x_ego, y_ego[i] - shift_y_ego])
        ax.add_patch(patch_ego)

        if adversary_name is not None:
            # update adversary
            rotation_angle_adversary = (yaw_adversary[i] - 90) % 360
            '''if car adversary: patch_adversary.set_angle(rotation_angle_adversary) '''
            shift_angle_adversary = rotation_angle_adversary - 180 / np.pi * np.arctan(car_width / car_length)
            shift_x_adversary = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(shift_angle_adversary * np.pi / 180)
            shift_y_adversary = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(shift_angle_adversary * np.pi / 180)
            '''if car adversary: patch_adversary.set_xy([x_adversary[i] + shift_x_adversary, y_adversary[i] - shift_y_adversary])
            ax.add_patch(patch_adversary) '''
            # updating adversary
            circle_adversary.center = x_adversary[i], y_adversary[i]
            ax.add_patch(circle_adversary)


        # updating pedestrians
        for key, pedestrian in enumerate(pedestrians_names):
            x_pedestrian = x_pedestrians[key]
            y_pedestrian = y_pedestrians[key]
            circle_pedestrian = circles_pedestrians[key]
            circle_pedestrian.center = x_pedestrian[i], y_pedestrian[i]
            ax.add_patch(circle_pedestrian)

        # updating vehicles
        for key, vehicle in enumerate(vehicles_names):
            x_vehicle = x_vehicles[key]
            y_vehicle = y_vehicles[key]
            yaw_vehicle = yaws_vehicles[key]
            patch_vehicle = patches_vehicles[key]

            rotation_angle_vehicle = (yaw_vehicle[i] - 90) % 360
            patch_vehicle.set_angle(rotation_angle_vehicle)
            shift_angle_vehicle = rotation_angle_vehicle - 180 / np.pi * np.arctan(car_width / car_length)
            shift_x_vehicle = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.sin(
                shift_angle_vehicle * np.pi / 180)
            shift_y_vehicle = 0.5 * np.sqrt(car_width ** 2 + car_length ** 2) * np.cos(
                shift_angle_vehicle * np.pi / 180)
            patch_vehicle.set_xy([x_vehicle[i] + shift_x_vehicle, y_vehicle[i] - shift_y_vehicle])
            ax.add_patch(patch_vehicle)
        return

    if "sampling_rate" in simout.otherParams:
        recorded_fps = simout.otherParams["sampling_rate"]
    else:
        recorded_fps = 100
    ani = FuncAnimation(fig, update, frames=len(simout.times[0::skip]))
    writer = PillowWriter(fps=recorded_fps)
    ani.save(str(savePath) + str(fileName) + ".gif", writer=writer)

    plt.clf()
    plt.close(fig)
    
