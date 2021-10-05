from typing import Optional, Tuple, Union


def process_play_post_game(play: dict, teams: dict, game_stats: dict):
    """Update post-game statistics (`game_stats`) by processing `play`

    Parameters
    ----------
    play : dict
        particular play to be processed
    teams : dict
        info about playing teams (away and home)
    game_stats : dict
        stats - continuously incremented play by play

    Raises
    ------
    ValueError
        Unknown type of play
    """
    # Determine active team
    if play["team"]["id"] == teams["away"]["id"]:
        team = "away"
    else:
        team = "home"

    # Determine type of the stat
    if play["type"] == "GOAL":
        stat = "G"
    elif play["type"] == "SHOT":
        stat = "SOG"
    elif play["type"] == "MISSED_SHOT":
        stat = "SMISS"
    elif play["type"] == "HIT":
        stat = "HIT"
    elif play["type"] == "TAKEAWAY":
        stat = "TAKE"
    elif play["type"] == "GIVEAWAY":
        stat = "GIVE"
    elif play["type"] == "BLOCKED_SHOT":
        stat = "BLK"
    elif play["type"] == "FACEOFF":
        stat = "FOW"
    elif play["type"] == "PENALTY":
        stat = "PPO"
        team = "away" if team == "home" else "home"  # switch team (penalty on vs. powerplay opportunity)
    else:
        raise ValueError(f"W: Unknown type of play: {play['type']}")

    # Determine strength situation (even, pp, sh)
    if play["strength_active"] == play["strength_opp"]:
        situation = "EV"
    elif play["strength_active"] > play["strength_opp"]:
        situation = "PP"
    else:
        situation = "SH"

    # Increase the stat
    if stat != "PPO":
        game_stats[f"{team}_{stat}_{situation}"] += 1
        game_stats[f"{team}_{stat}_ALL"] += 1
    else:
        game_stats[f"{team}_PPO"] += 1


def get_opp_situation(situation: str) -> Optional[str]:
    """Convert situation to situation of opposite team.

    If active team has power play (PP), opposite team is short-handed (SH) and vice versa.

    Parameters
    ----------
    situation : str
        situation of active team

    Returns
    -------
    Optional[str]
        Situation of the opposite team. In other situations, None is returned
    """
    if situation == "SH":
        return "PP"
    if situation == "PP":
        return "SH"
    return None


def get_opp_team(active_team: str) -> Optional[str]:
    """Convert active team to opposite team.

    If active team is a home team, opposite team is away and vice versa.

    Parameters
    ----------
    active_team : str
        active team

    Returns
    -------
    Optional[str]
        opposite team; None if `active team` is not "away" nor "home".
    """
    if active_team == "away":
        return "home"
    if active_team == "home":
        return "away"
    return None


def determine_winner(goal_diff: int) -> str:
    """Choose winner from goal difference.

    Parameters
    ----------
    goal_diff : int
        goal difference: `away_G_ALL` - `home_G_ALL`

    Returns
    -------
    str
        ["away"|"home"|"draw"]
    """
    if goal_diff > 0:
        return "away"
    if goal_diff < 0:
        return "home"
    return "draw"


def get_source_col_names(
    for_against: str, situation: str, active_team: str, stat: str, opp_situation: str, opp_team: str
) -> Union[str, Tuple[str]]:
    """Construct name of a column in post-game stats dataset.

    Parameters
    ----------
    for_against : str
        for or agains with respect to active team
    situation : str
        situation (PP, SH, EV, ALL)
    active_team : str
        active team (home, away or both previous games of the team)
    stat : str
        statistic name
    opp_situation : str
        situation of the opposite team
    opp_team : str
        opposite team (home or away)

    Returns
    -------
    Union[str, Tuple[str]]
        name of the desired column or two names if both variant
    """
    if active_team in ["away", "home"]:
        if for_against == "F":
            return f"{active_team}_{stat}_{situation}"
        if situation in ["PP", "SH"]:
            return f"{opp_team}_{stat}_{opp_situation}"
        return f"{opp_team}_{stat}_{situation}"
    else:  # Both - return two column names
        if for_against == "F":
            return f"away_{stat}_{situation}", f"home_{stat}_{situation}"
        if situation in ["PP", "SH"]:
            return f"home_{stat}_{opp_situation}", f"away_{stat}_{opp_situation}"
        return f"home_{stat}_{situation}", f"away_{stat}_{situation}"
