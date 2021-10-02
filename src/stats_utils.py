def process_play_post_game(play, teams, game_stats): # noqa C901
    """
    Update post-game statistics by processing `play`.

    :param play:
    :param game_stats:
    :return:
    """
    # Determine active team
    if play['team']['id'] == teams['away']['id']:
        team = "away"
    else:
        team = "home"

    # Determine type of the stat
    if play['type'] == "GOAL":
        stat = "G"
    elif play['type'] == "SHOT":
        stat = "SOG"
    elif play['type'] == "MISSED_SHOT":
        stat = "SMISS"
    elif play['type'] == "HIT":
        stat = "HIT"
    elif play['type'] == "TAKEAWAY":
        stat = "TAKE"
    elif play['type'] == "GIVEAWAY":
        stat = "GIVE"
    elif play['type'] == "BLOCKED_SHOT":
        stat = "BLK"
    elif play['type'] == "FACEOFF":
        stat = "FOW"
    elif play['type'] == "PENALTY":
        stat = "PPO"
        team = "away" if team == "home" else "home"     # switch team (penalty on vs. powerplay opportunity)
    else:
        raise ValueError(f"W: Unknown type of play: {play['type']}")

    # Determine strength situation (even, pp, sh)
    if play['strength_active'] == play['strength_opp']:
        situation = "EV"
    elif play['strength_active'] > play['strength_opp']:
        situation = "PP"
    else:
        situation = "SH"

    # Increase the stat
    if stat != "PPO":
        game_stats[f"{team}_{stat}_{situation}"] += 1
        game_stats[f"{team}_{stat}_ALL"] += 1
    else:
        game_stats[f"{team}_PPO"] += 1


def get_opp_situation(situation):
    if situation == "SH":
        return "PP"
    if situation == "PP":
        return "SH"
    return None


def get_opp_team(active_team):
    if active_team == "away":
        return "home"
    if active_team == "home":
        return "away"
    return None


def determine_winner(goal_diff):
    if goal_diff > 0:
        return "away"
    if goal_diff < 0:
        return "home"
    return "draw"


def get_source_col_names(for_against, situation, active_team, stat, opp_situation, opp_team):
    if active_team in ['away', 'home']:
        if for_against == "F":
            return f"{active_team}_{stat}_{situation}"
        if situation in ['PP', 'SH']:
            return f"{opp_team}_{stat}_{opp_situation}"
        return f"{opp_team}_{stat}_{situation}"
    else:   # Both - return two column names
        if for_against == "F":
            return f"away_{stat}_{situation}", f"home_{stat}_{situation}"
        if situation in ['PP', 'SH']:
            return f"home_{stat}_{opp_situation}", f"away_{stat}_{opp_situation}"
        return f"home_{stat}_{situation}", f"away_{stat}_{situation}"
