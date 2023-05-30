import math

import numpy as np

from .constants import GRAVITATIONAL_CONSTANT


def _kinematic_viscosity(temperature):
    """
    The kinematic viscosity barely changes with pressure. The below polynomial
    fit has been deduced using the `iapws` package, which implements the IAPWS
    standard. The valid range of temperatures is 20 - 130 C.

    Below the snippet of code used to get the polynomial coefficients
    (possibly useful when refitting):

    .. code-block:: python

        from iapws import IAPWS95

        import numpy as np

        res = []
        for t in np.linspace(20, 130, 1000):
            res.append((t, IAPWS95(T=273.15 + t, P=0.5).nu))
        print(np.polyfit(*zip(*res), 4))
    """

    if temperature < 20 or temperature > 130:
        raise Exception(
            "Temperature should be in the range 20 - 130 °C.\n"
            "Note that we use Celcius as the unit, not Kelvin."
        )

    return np.polyval(
        [7.53943453e-15, -3.01485854e-12, 4.75924986e-10, -3.79135487e-08, 1.58737429e-06],
        temperature,
    )


def _colebrook_white(reynolds, relative_roughness, friction_factor=0.015):
    for _ in range(1000):
        friction_factor_old = friction_factor

        reynolds_star = (
            1 / math.sqrt(8.0) * reynolds * math.sqrt(friction_factor) * relative_roughness
        )
        friction_factor = (
            1.0
            / (
                -2.0
                * math.log10(
                    2.51 / reynolds / math.sqrt(friction_factor) * (1 + reynolds_star / 3.3)
                )
            )
            ** 2
        )

        if (
            abs(friction_factor - friction_factor_old) / max(friction_factor, friction_factor_old)
            < 1e-6
        ):
            return friction_factor
    else:
        raise Exception("Colebrook-White did not converge")


def friction_factor(velocity, diameter, wall_roughness, temperature):
    """
    Darcy-weisbach friction factor calculation from both laminar and turbulent
    flow.
    """

    kinematic_viscosity = _kinematic_viscosity(temperature)
    reynolds = velocity * diameter / kinematic_viscosity

    assert velocity >= 0

    if velocity == 0.0 or diameter == 0.0:
        return 0.0
    elif reynolds <= 2000.0:
        friction_factor = 64.0 / reynolds
    elif reynolds >= 4000.0:
        friction_factor = _colebrook_white(reynolds, wall_roughness / diameter)
    else:
        fac_turb = _colebrook_white(4000.0, wall_roughness / diameter)
        fac_laminar = 64.0 / 2000.0
        w = (reynolds - 2000.0) / 2000.0
        friction_factor = w * fac_turb + (1 - w) * fac_laminar

    return friction_factor


def head_loss(velocity, diameter, length, wall_roughness, temperature):
    """
    Head loss for a circular pipe of given length.
    """

    f = friction_factor(velocity, diameter, wall_roughness, temperature)

    return length * f / (2 * GRAVITATIONAL_CONSTANT) * velocity**2 / diameter


def get_linear_pipe_dh_vs_q_fit(
    diameter, length, wall_roughness, temperature, n_lines=10, v_max=2.5
):
    area = math.pi * diameter**2 / 4

    v_points = np.linspace(0.0, v_max, n_lines + 1)
    q_points = v_points * area

    h_points = np.array(
        [head_loss(v, diameter, length, wall_roughness, temperature) for v in v_points]
    )

    a = np.diff(h_points) / np.diff(q_points)
    b = h_points[1:] - a * q_points[1:]

    return a, b


def get_linear_pipe_power_hydraulic_vs_q_fit(
    rho, diameter, length, wall_roughness, temperature, n_lines=10, v_max=2.5
):
    """
    power_hydraulic = b + (a * Q)
    """
    area = math.pi * diameter**2 / 4.0

    v_points = np.linspace(0.0, v_max, n_lines + 1)
    q_points = v_points * area
    power_hydraulic_points = np.array(
        [
            rho
            * GRAVITATIONAL_CONSTANT
            * abs(head_loss(v, diameter, length, wall_roughness, temperature))
            * v
            * area
            for v in v_points
        ]
    )

    a = np.diff(power_hydraulic_points) / np.diff(q_points)  # calc gradients for n_line segments
    b = power_hydraulic_points[1:] - a * q_points[1:]

    return a, b
