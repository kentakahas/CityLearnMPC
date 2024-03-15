# type hinting
from typing import List, Tuple

# Data manipulation
from datetime import datetime, timedelta


def set_schema_buildings(
        schema: dict, building_index: list
) -> Tuple[dict, List[str]]:
    """Select buildings to set as active in the schema.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    building_index: list
        List of building indices to include in analysis

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active buildings set.
    buildings: List[str]
        List of selected buildings.
    """

    # get all building names
    buildings = list(schema['buildings'].keys())

    # remove buildings 12 and 15 as they have peculiarities in their data
    # that are not relevant to this tutorial
    buildings_to_exclude = ['Building_12', 'Building_15']

    for b in buildings_to_exclude:
        buildings.remove(b)

    buildings = [f'Building_{i}' for i in building_index]

    # update schema to only included selected buildings
    for b in schema['buildings']:
        if b in buildings:
            schema['buildings'][b]['include'] = True
        else:
            schema['buildings'][b]['include'] = False

    return schema, buildings


def get_timestep(
    month: int, day: int, hour: int = 0
) -> int:
    dt = datetime(2017, month, day, hour)

    if month < 8:
        time_step = dt.timetuple().tm_yday * 24 + hour + 3648
    else:
        time_step = dt.timetuple().tm_yday * 24 + hour - 5112
    
    return time_step


def get_datetime(timestep: int) -> datetime:
    return datetime(2016, 8, 1) + timedelta(hours=timestep)


def set_schema_simulation_period(
    schema: dict, start: int, stop: int
) -> dict:
    """Select environment simulation start and end time steps.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    start: int
        Start timestep.
    stop: int
        Stop timestep.

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with `simulation_start_time_step`
        and `simulation_end_time_step` key-values set.
    """

    # update schema simulation time steps
    schema['simulation_start_time_step'] = start
    schema['simulation_end_time_step'] = stop

    return schema


def set_active_observations(
    schema: dict, active_observations: List[str]
) -> dict:
    """Set the observations that will be part of the environment's
    observation space that is provided to the control agent.

    Parameters
    ----------
    schema: dict
        CityLearn dataset mapping used to construct environment.
    active_observations: List[str]
        Names of observations to set active to be passed to control agent.

    Returns
    -------
    schema: dict
        CityLearn dataset mapping with active observations set.
    """

    active_count = 0

    for o in schema['observations']:
        if o in active_observations:
            schema['observations'][o]['active'] = True
            active_count += 1
        else:
            schema['observations'][o]['active'] = False

    valid_observations = list(schema['observations'].keys())
    assert active_count == len(active_observations), \
        'the provided observations are not all valid observations.'\
        f' Valid observations in CityLearn are: {valid_observations}'

    return schema
