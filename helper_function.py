"""
This is a helper function that can be used to validate that the actions you want to take
are physically possible for the battery.

For example:
- If the battery has 2 MWh of energy, you cannot discharge 3 MWh.
- If the battery has 8 MWh of energy, you cannot charge it with another 3 MWh
  without exceeding its capacity.
"""


def validate_battery_actions(
    actions,
    capacity=10,
    initial_soc=5,
    timestep_hours=1/4,
    return_trace=False,
    reset_daily=True,
):
    """
    Validate a sequence of battery charge/discharge actions for a daily-reset battery model.

    Parameters
    ----------
    actions : list[float]
        Sequence of charge/discharge actions in MWh.
        Positive values represent charging; negative values represent discharging.
        actions[0] corresponds to 00:00 of day 1.

    capacity : float, default=10
        Energy capacity of the battery in MWh. The state of charge (SoC) must remain within [0, capacity].

    initial_soc : float, default=5
        The SoC at the beginning of each day (00:00).
        If `reset_daily=True`, the SoC is reset to this value at the start of every day.

    timestep_hours : float, default=0.25
        Duration of each action interval in hours.
        Used only to infer the number of intervals per day for daily resets
        (24 / timestep_hours).

    return_trace : bool, default=False
        If True, the function returns the full SoC trajectory across all time steps.

    reset_daily : bool, default=True
        If True, the SoC is automatically reset to `initial_soc` at the start of each new day
        (i.e., before processing steps 96, 192, 288, ... for 15-minute intervals).

    Behavior & Assumptions
    ----------------------
    - Each action is an energy quantity (MWh) applied during that interval.
    - SoC evolves (for valid actions) as: SoC[t+1] = SoC[t] + action[t].
    - If an action would make SoC < 0 or > capacity, that action is:
        * NOT applied (SoC stays the same for that step)
        * recorded as a warning
    - Daily resets occur at midnight *before* applying the first action of the new day.
      Example with 15-minute intervals:
          Day 1: steps 0-95
          Reset before step 96
          Day 2: steps 96-191
    - The length of `actions` may span multiple days (e.g., 96 * 7 for a full week).

    Returns
    -------
    (bool, list[str]) or (bool, list[float], list[str])
        - If return_trace is False:
              (is_valid, warnings)
        - If return_trace is True:
              (is_valid, soc_trace, warnings)

        is_valid  : True if all actions were valid, False if at least one was invalid.
        soc_trace : List of SoC values over time (only when return_trace=True).
        warnings  : List of warning strings for all invalid actions.
    """

    soc = initial_soc
    soc_trace = [soc]
    warnings = []

    intervals_per_day = int(round(24 / timestep_hours))

    for i, a in enumerate(actions):
        if reset_daily and i > 0 and (i % intervals_per_day == 0):
            soc = initial_soc

        soc_new = soc + a


        if soc_new < 0:
            warnings.append(
                f"Step {i}: discharge too large. "
                f"Requested action {a} MWh, SoC would become {soc_new} < 0. "
                "Action ignored."
            )
            soc_trace.append(soc)
            continue

        if soc_new > capacity:
            warnings.append(
                f"Step {i}: charge too large. "
                f"Requested action {a} MWh, SoC would become {soc_new} > {capacity}. "
                "Action ignored."
            )
            soc_trace.append(soc)
            continue

        soc = soc_new
        soc_trace.append(soc)

    is_valid = len(warnings) == 0

    if return_trace:
        return is_valid, soc_trace, warnings

    return is_valid, warnings
