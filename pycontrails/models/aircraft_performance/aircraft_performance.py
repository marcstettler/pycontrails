"""Ideas for aircraft performance abstraction.

WIP
"""

import abc
from typing import MutableMapping
import numpy as np

from pycontrails.core.flight import Flight
from pycontrails.core.models import Model, ModelParams, interpolate_met


class AircraftPerformanceModelParams(ModelParams):
    """Parameters for :class:`AircraftPerformanceModel`."""

    #: Hierarchies of models used to query for given aircraft type. For example,
    #: with `model=["openap", "bada3"]`, first the "openap" model would be checked
    #: for the provided aircraft type. The "bada3" model would only be used if
    #: the aircraft type is NOT available in "openap".
    model: str | list[str]


class AircraftPerformanceModel(Model):
    def eval(self, source: Flight | list[Flight], **params):
        self.update_params(params)
        self.set_source(source)
        self.source = self.require_source_type(Flight)
        self.downselect_met()
        self.set_source_met()

        # calculate true airspeed if not included in data
        if "true_airspeed" not in self.source:

            # Two step fallback: try to find u_wind and v_wind.
            try:
                u = interpolate_met(self.met, self.source, "eastward_wind", **self.interp_kwargs)
                v = interpolate_met(self.met, self.source, "northward_wind", **self.interp_kwargs)

            except (ValueError, KeyError):
                raise ValueError(
                    "Variable `true_airspeed` not found. Include 'eastward_wind' and 'northward_wind' variables on `met`"
                    "in model constructor, or define `true_airspeed` data on flight. "
                    "This can be achieved by calling the `Flight.segment_true_airspeed` method."
                )

            self.source["true_airspeed"] = self.source.segment_true_airspeed(u, v)

        # TODO: allow aircraft engine to be defined separate from aircraft type?
        aircraft_type = self.source.attrs["aircraft_type"]
        bada = get_bada(
            aircraft_type,
            bada3_path=self.params["bada3_path"],
            bada4_path=self.params["bada4_path"],
            bada_priority=self.params["bada_priority"],
        )
        aircraft_engine = bada.get_aircraft_engine_properties(aircraft_type)

        # set flight attributes based on engine, if they aren't already defined
        self.source.attrs.setdefault("bada_model", type(bada).__name__)
        self.source.attrs.setdefault("aircraft_type_bada", aircraft_engine.atyp_bada)
        self.source.attrs.setdefault("wingspan", aircraft_engine.wing_span)
        self.source.attrs.setdefault("max_mach", aircraft_engine.max_mach_num)
        self.source.attrs.setdefault("max_altitude", units.ft_to_m(aircraft_engine.max_altitude_ft))
        self.source.attrs.setdefault("engine_name", aircraft_engine.engine_name)
        self.source.attrs.setdefault("n_engine", aircraft_engine.n_engine)

        outputs = bada.simulate_fuel_and_performance(
            atyp_icao=aircraft_type,
            altitude_ft=self.source.altitude_ft,
            time=self.source["time"],
            true_airspeed=self.source["true_airspeed"],
            air_temperature=self.source["air_temperature"],
            q_fuel=self.source.fuel.q_fuel,
            correct_fuel_flow=self.params["correct_fuel_flow"],
            model_choice=self.params["model_choice"],
            load_factor=self.get_source_param("load_factor", None),
            n_iter=self.params["n_iter"],
            aircraft_mass=self.get_source_param("aircraft_mass", None),
            engine_efficiency=self.get_source_param("engine_efficiency", None),
            fuel_flow=self.get_source_param("fuel_flow", None),
            thrust=self.get_source_param("thrust", None),
        )

        # set outputs to flight, don't overwrite
        for output_var in [
            "aircraft_mass",
            "engine_efficiency",
            "fuel_flow",
            "fuel_burn",
            "thrust",
            "rocd",
        ]:
            self.source.setdefault(output_var, getattr(outputs, output_var))

        self._cleanup_indices()

        self.source.attrs["total_fuel_burn"] = np.nansum(outputs.fuel_burn)

        return self.source


@abc.ABC
class AircraftPerformance(MutableMapping):
    """Support for loading static aircraft data and simulating aircraft performance."""

    def check_availability(self, aircraft_type: str):
        pass

    def load_data(self):
        pass

    def simulate_once(
        self,
        atyp_icao: str,
        altitude_ft: np.ndarray,
        time: np.ndarray,
        true_airspeed: np.ndarray,
        air_temperature: np.ndarray,
        q_fuel: float,
        correct_fuel_flow: bool,
        model_choice: str,
        load_factor: None | float = None,
        n_iter: int = 5,
        aircraft_mass: np.ndarray | float | None = None,
        engine_efficiency: np.ndarray | float | None = None,
        fuel_flow: np.ndarray | float | None = None,
        thrust: np.ndarray | float | None = None,
    ):
        pass

    def iterate(
        self,
    ):
        pass

    def nominal(self, aircraft_type: str):
        pass
