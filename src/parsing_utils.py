import numpy as np
import pandas as pd
from typing import Iterable, Union


def add_strength(game, filtered):
    """
    TODO
    :param game:
    :param filtered:
    :return:
    """
    return filtered


def parse_game(game: dict) -> Iterable[dict]:
    game_shots = []
    for i_event in range(len(game['plays'])):
        prev_event = game['plays'][i_event - 1]
        curr_event = game['plays'][i_event]
        if curr_event['type'] in ["GOAL", "SHOT", "MISSED_SHOT"]:
            game_shots.append(parse_event(curr_event, prev_event, game['teams']))
    return game_shots


def parse_event(curr_event: dict, prev_event: dict, teams: dict) -> dict:
    result = dict()

    # Shot type
    if 'shotType' in curr_event:
        result['shot_type'] = curr_event['shotType']
    else:
        result['shot_type'] = np.nan

    # Distance and angle from goal
    goal_pos = determine_goal_position(curr_event, teams)
    curr_dist = get_distance_from_goal(curr_event, goal_pos)
    curr_angle = get_angle_to_goal(curr_event, goal_pos)
    result['distance'] = curr_dist
    result['angle'] = curr_angle

    # Change from previous event
    prev_dist = get_distance_from_goal(prev_event, goal_pos)
    prev_angle = get_angle_to_goal(prev_event, goal_pos)
    result['distance_change'] = curr_dist - prev_dist
    result['angle_change'] = curr_angle - prev_angle
    result['time_change'] = get_seconds_from_time(curr_event['time']) - get_seconds_from_time(prev_event['time'])
    result['prev_event_type'] = prev_event['type']
    result['prev_event_same_team'] = 1 if curr_event['team']['id'] == prev_event['team']['id'] else 0

    # Home team advantage
    result['is_home'] = 1 if curr_event['team']['id'] == teams['home']['id'] else 0

    # Current score
    result['goal_diff'] = get_goal_diff(curr_event, prev_event, teams)

    # Outcome
    result['outcome'] = 1 if curr_event['type'] == "GOAL" else 0
    return result


def determine_goal_position(event:dict, teams: dict) -> np.ndarray:
    """
    Determines location of opponent's goal (from a shot/goal-like event) based on the period and shooting team.

    Goal positions (89, 0) and (-89, 0) are just approximate. There were some issues about validity of this assumption
    discussed on Twitter.

    :param event:
    :param teams:
    :return:
    """
    goal_pos = np.array([-89.0, 0.0])               # approximate location (there are some issues discussed about it on Twitter)
    if event['team']['id'] == teams['home']['id']:  # home team is the active one in this event
        if event['period'] % 2 == 0:                # 2nd period and overtime (home: left -> right)
            goal_pos = -goal_pos
    else:
        if event['period'] % 2 == 1:                # 1st and 3rd period (away: left -> right)
            goal_pos = -goal_pos
    return goal_pos


def get_distance_from_goal(event: dict, goal_coords: np.ndarray) -> Union[np.float32, np.ndarray]:
    """
    Calculates distance of given event from the given goal.

    :param event: dict - event with coordinates
    :param goal_coords: np.ndarray - coordinates of goal
    :return: distance of given event and given
    """

    x = event['coordinates']['x'] if 'x' in event['coordinates'] else 0.0
    y = event['coordinates']['y'] if 'y' in event['coordinates'] else 0.0
    event_coords = np.array([x, y])
    return np.linalg.norm(event_coords - goal_coords)


def get_angle_to_goal(event: dict, goal_coords: np.ndarray) -> np.float32:
    """
    Calculates angle of the given event to the given goal (in radians).

    :param event: TODO
    :param goal_coords: TODO
    :return: TODO
    """
    x = event['coordinates']['x'] if 'x' in event['coordinates'] else 0.0
    y = event['coordinates']['y'] if 'y' in event['coordinates'] else 0.0
    event_coords = np.array([x, y])
    if goal_coords[0] < 0:                  # left goal
        goal_line_point = np.array([0, -10])
    else:                                   # right goal
        goal_line_point = np.array([0, 10])
    goal_line_vec = (goal_coords + goal_line_point) - goal_coords
    event_vec = event_coords - goal_coords
    goal_line_vec /= np.linalg.norm(goal_line_vec)
    event_vec /= np.linalg.norm(event_vec)
    return np.arccos(np.dot(goal_line_vec, event_vec))


def get_seconds_from_time(time):
    m, s = time.split(":")
    return int(m)*60 + int(s)


def get_goal_diff(curr_event, prev_event, teams):
    if curr_event['type'] == "GOAL":
        diff = prev_event['score']['home'] - prev_event['score']['away']
    else:
        diff = curr_event['score']['home'] - prev_event['score']['away']

    if curr_event['team']['id'] == teams['away']['id']:
        diff = -diff
    return diff
