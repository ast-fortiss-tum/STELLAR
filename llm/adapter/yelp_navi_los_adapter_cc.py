from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from collections.abc import Mapping

from examples.car_control.models_new import CCContentOutput


def _get_in(d: Any, path: Tuple[str, ...], default=None):
    """
    Traverse nested POI where each level may be:
      - dict-like (Mapping): key access
      - pydantic / object-like: attribute access
    This fixes the original implementation which only supported dicts.
    """
    cur: Any = d
    for p in path:
        if cur is None:
            return default

        # dict-like
        if isinstance(cur, Mapping):
            if p not in cur:
                return default
            cur = cur[p]
            continue

        # object-like (pydantic)
        if hasattr(cur, p):
            cur = getattr(cur, p)
            continue

        return default

    return cur


def _to_onoff(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("on", "off"):
            return s
    return None


def _to_window_state(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("open", "closed", "ajar"):
            return s
    return None


def _to_headlights_mode(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("off", "on", "auto"):
            return s
    return None


def _to_seat_heating_level(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("off", "low", "medium", "high", "1", "2", "3"):
            return s
    if isinstance(v, (int, float)) and int(v) == v:
        if int(v) in (0, 1, 2, 3):
            return str(int(v))
    return None


def _reading_lights_poi_path(position: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not position:
        return None
    p = position.strip().lower()
    if p in ("driver", "passenger", "front", "front_left", "front_right"):
        return ("lights", "interior_front")
    if p in ("rear", "rear_left", "rear_right", "back"):
        return ("lights", "interior_rear")
    return None


def _fog_lights_poi_path(fog_light_position: Optional[str]) -> Optional[Tuple[str, ...]]:
    return ("lights", "fog_lights")


def _window_poi_path(position: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not position:
        return None
    p = position.strip().lower()
    mapping = {
        "front_left": ("windows", "front_left"),
        "front_right": ("windows", "front_right"),
        "rear_left": ("windows", "rear_left"),
        "rear_right": ("windows", "rear_right"),
    }
    return mapping.get(p)


def _seat_heating_poi_path(position: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not position:
        return None
    p = position.strip().lower()
    mapping = {
        "driver": ("seat_heating", "driver"),
        "passenger": ("seat_heating", "front_passenger"),
        "front_passenger": ("seat_heating", "front_passenger"),
    }
    return mapping.get(p)


def convert_yelp_navi_los_to_content_output(
    yelp_poi: Dict[str, Any],
    content_input: Dict[str, Any],
) -> Dict[str, Any]:
    """
    FIXES (without changing function inputs / names):
      - CCContentOutput is a pydantic object: you cannot do out["x"] or out.keys().
        Use attribute assignment + model_fields (pydantic v2) / __fields__ (v1).
      - _get_in now supports both dict POI and pydantic POI objects.
      - Keeps your original behavior: maps both *_target passthrough and *_initial from POI.
    """
    poi_retrieved = yelp_poi

    # pydantic output object
    out = CCContentOutput()

    # content_input may be dict or pydantic; use getattr fallback
    system = content_input.get("system") if isinstance(content_input, dict) else getattr(content_input, "system", None)
    position = content_input.get("position") if isinstance(content_input, dict) else getattr(content_input, "position", None)

    # Set tracked component identifiers
    if hasattr(out, "system"):
        out.system = system
    if hasattr(out, "position"):
        out.position = position

    # Pass-through all *_target fields that exist on CCContentOutput AND are present in content_input
    # Support both Pydantic v2 and v1
    if hasattr(out, "model_fields"):  # pydantic v2
        out_field_names = set(out.model_fields.keys())
    elif hasattr(out, "__fields__"):  # pydantic v1
        out_field_names = set(out.__fields__.keys())
    else:
        out_field_names = set()

    for fname in out_field_names:
        if not fname.endswith("_target"):
            continue

        if isinstance(content_input, dict):
            if fname in content_input:
                setattr(out, fname, content_input.get(fname))
        else:
            if hasattr(content_input, fname):
                setattr(out, fname, getattr(content_input, fname))

    # Initial-state mapping from POI (unchanged logic, but using attribute assignment)
    if system == "reading_lights":
        poi_path = _reading_lights_poi_path(position)
        poi_val = _get_in(poi_retrieved, poi_path) if poi_path else None
        if hasattr(out, "onoff_state_target"):
            out.onoff_state_target = _to_onoff(poi_val)

    elif system == "windows":
        poi_path = _window_poi_path(position)
        poi_val = _get_in(poi_retrieved, poi_path) if poi_path else None
        if hasattr(out, "window_state_target"):
            out.window_state_target = _to_window_state(poi_val)

    elif system == "fog_lights":
        fog_pos = content_input.get("fog_light_position") if isinstance(content_input, dict) else getattr(content_input, "fog_light_position", None)
        if hasattr(out, "fog_light_position"):
            out.fog_light_position = fog_pos

        # poi_val = _get_in(poi_retrieved, _fog_lights_poi_path(fog_pos))
        # if hasattr(out, "onoff_state_initial"):
        #     out.onoff_state_initial = _to_onoff(poi_val)

    elif system in ("head_lights", "headlights"):
        poi_val = _get_in(poi_retrieved, ("lights", "headlights"))
        if hasattr(out, "head_lights_mode_target"):
            out.head_lights_mode_target = _to_headlights_mode(poi_val)

    elif system in ("climate_temperature", "climate"):
        poi_val = _get_in(poi_retrieved, ("climate", "temperature_c"))
        if hasattr(out, "climate_temperature_value_target"):
            out.climate_temperature_value_target = poi_val if isinstance(poi_val, (int, float)) else None

    elif system == "seat_heating":
        poi_path = _seat_heating_poi_path(position)
        poi_val = _get_in(poi_retrieved, poi_path) if poi_path else None
        if hasattr(out, "seat_heating_level_target"):
            out.seat_heating_level_target = _to_seat_heating_level(poi_val)

    # Remove interactive blocking input() (this will hang in services/tests)
    # Keep prints if you want, but they can be noisy; leaving them as-is would be okay.
    # print("content input:", content_input)
    # print("poi retrieved:", poi_retrieved)
    # print("content output after mapping:", out)

    # Return type annotation says Dict[str, Any], but you were returning the pydantic object.
    # To avoid changing signature, return a dict representation if available.

    print("out:", out)
    # if hasattr(out, "model_dump"):  # pydantic v2
    #     return out.model_dump()
    # if hasattr(out, "dict"):  # pydantic v1
    #     return out.dict()
    return out  # type: ignore[return-value]

# def convert_yelp_navi_los_to_content_output(yelp_poi: dict, content_input: CCContentInput) -> CCContentOutput:
#     output = CCContentOutput()
#     print("content input:", content_input)
#     print("yelp_poi:", yelp_poi)

#     window_position_map = {
#         "driver": "front_left",
#         "passenger": "front_right",
#         "back_left": "rear_left",
#         "back_right": "rear_right",
#     }

#     seat_position_map = {
#         "driver": "driver",
#         "front_passenger": "front_passenger",
#         "back_left": "back_left",
#         "back_center": "back_center",
#         "back_right": "back_right",
#     }

#     system = content_input.system
#     if system is None:
#         return output  # nothing requested, so keep everything None

#     # ----------------
#     # WINDOWS
#     # ----------------
#     if system == "windows":
#         windows = yelp_poi.get("windows")
#         if not isinstance(windows, dict):
#             return output

#         chosen_pos = content_input.position or "driver"
#         yelp_key = window_position_map.get(chosen_pos)
#         if not yelp_key or yelp_key not in windows:
#             return output

#         state = windows.get(yelp_key)  # "closed" / "open"
#         if state not in ("closed", "open"):
#             return output

#         output.system = "windows"
#         output.position = chosen_pos
#         output.window_state_target = "close" if state == "closed" else "open"
#         return output

#     # ----------------
#     # HEAD LIGHTS
#     # ----------------
#     if system == "head_lights":
#         lights = yelp_poi.get("lights", {})
#         if not isinstance(lights, dict):
#             return output

#         hl = lights.get("headlights")  # your input: "off"
#         # CCContentOutput.head_lights_mode_target does NOT allow "off" (only parking/low/high/auto)
#         if hl in ("parking", "low_beam", "high_beam", "auto"):
#             output.system = "head_lights"
#             output.head_lights_mode_target = hl
#         # if "off" -> leave empty (no command)
#         return output

#     # ----------------
#     # FOG LIGHTS
#     # ----------------
#     if system == "fog_lights":
#         lights = yelp_poi.get("lights", {})
#         if not isinstance(lights, dict):
#             return output

#         fog = lights.get("fog_lights")  # "on"/"off"
#         if fog not in ("on", "off"):
#             return output

#         output.system = "fog_lights"
#         output.fog_light_position = content_input.fog_light_position or "front"
#         output.onoff_state_target = fog
#         return output

#     # ----------------
#     # AMBIENT LIGHTS
#     # ----------------
#     if system == "ambient_lights":
#         lights = yelp_poi.get("lights", {})
#         if not isinstance(lights, dict):
#             return output

#         amb = lights.get("ambient")  # in your input: "low"
#         if amb is None:
#             return output

#         # CCContentOutput.onoff_state_target only allows on/off, so collapse brightness -> on
#         mapped = "off" if amb == "off" else "on"

#         output.system = "ambient_lights"
#         output.onoff_state_target = mapped
#         return output

#     # ----------------
#     # CLIMATE (temperature only)
#     # ----------------
#     if system == "climate":
#         climate = yelp_poi.get("climate")
#         if not isinstance(climate, dict):
#             return output

#         temp = climate.get("temperature_c")
#         if temp is None:
#             return output

#         intval = int(round(float(temp)))
#         if 16 <= intval <= 26:
#             output.system = "climate"
#             output.climate_temperature_value_target = intval
#         return output

#     # ----------------
#     # SEAT HEATING
#     # ----------------
#     if system == "seat_heating":
#         seat_heating = yelp_poi.get("seat_heating")
#         if not isinstance(seat_heating, dict):
#             return output

#         chosen_seat = content_input.seat_position
#         if chosen_seat is None:
#             return output

#         yelp_key = seat_position_map.get(chosen_seat)
#         if not yelp_key or yelp_key not in seat_heating:
#             return output

#         state = seat_heating.get(yelp_key)  # "off"/"low"/"medium"/"high" or number
#         if state == "off":
#             level = 0
#         elif state == "low":
#             level = 1
#         elif state == "medium":
#             level = 2
#         elif state == "high":
#             level = 3
#         elif isinstance(state, (int, float)):
#             level = int(state)
#         else:
#             return output

#         if level not in (0, 1, 2, 3):
#             return output

#         output.system = "seat_heating"
#         output.seat_position = chosen_seat
#         output.seat_heating_level_target = level
#         return output

#     # Not implemented systems in your converter yet:
#     # - reading_lights
#     # - fan
#     print("output:", output)
#     return output

# def convert_yelp_navi_los_to_content_output(yelp_poi: dict, content_input: CCContentInput) -> CCContentOutput:
#     output = CCContentOutput()
#     #print("###############################################")
#     print("yelp_poi:", yelp_poi)
#     # Mapeo de posiciones a claves del diccionario yelp_poi
#     window_position_map = {
#         "driver": "front_left",
#         "passenger": "front_right",
#         "back_left": "rear_left",
#         "back_right": "rear_right",
#     }

#     seat_position_map = {
#         "driver": "driver",
#         "front_passenger": "front_passenger",
#         "back_left": "back_left",
#         "back_center": "back_center",
#         "back_right": "back_right",
#     }

#     # -------------------------
#     # Helper: procesar 1 system
#     # -------------------------
#     def fill_system(system_field, position_field, seat_position_field,
#                     fog_pos_field, target_window_field, target_onoff_field,
#                     target_headlights_field, target_temp_field,
#                     target_seat_heat_field,
#                     system_name, position, seat_position):

#         if system_name == "windows":
#             if "windows" in yelp_poi and position:
#                 key = window_position_map.get(position)
#                 if key and key in yelp_poi["windows"]:
#                     setattr(output, system_field, "windows")
#                     setattr(output, position_field, position)

#                     state = yelp_poi["windows"][key]  # "closed" o "open"
#                     mapped = "close" if state == "closed" else "open"
#                     setattr(output, target_window_field, mapped)
#             elif "windows" in yelp_poi:
#                 key = window_position_map.get("driver")
#                 if key and key in yelp_poi["windows"]:
#                     setattr(output, system_field, "windows")
#                     setattr(output, position_field, position)

#                     state = yelp_poi["windows"][key]  # "closed" o "open"
#                     mapped = "close" if state == "closed" else "open"
#                     setattr(output, target_window_field, mapped)

#         elif system_name == "head_lights":
#             if "lights" in yelp_poi:
#                 print(yelp_poi["lights"])
#                 hl = yelp_poi["lights"].get("headlights")
#                 if hl is not None:
#                     setattr(output, system_field, "head_lights")
#                     setattr(output, target_headlights_field, hl)

#         elif system_name == "fog_lights":
#             if "lights" in yelp_poi:
#                 fog = yelp_poi["lights"].get("fog_lights")
#                 if fog is not None:
#                     setattr(output, system_field, "fog_lights")
#                     setattr(output, fog_pos_field, "front")  # si no hay info
#                     setattr(output, target_onoff_field, fog)

#         elif system_name == "ambient_lights":
#             if "lights" in yelp_poi:
#                 amb = yelp_poi["lights"].get("ambient")
#                 if amb is not None:
#                     setattr(output, system_field, "ambient_lights")
#                     setattr(output, target_onoff_field, amb)

#         elif system_name == "climate":
#             if "climate" in yelp_poi:
#                 temp = yelp_poi["climate"].get("temperature_c")
#                 if temp:
#                     setattr(output, system_field, "climate")
#                     intval = int(round(temp))
#                     if 16 <= intval <= 26:
#                         setattr(output, target_temp_field, intval)

#         elif system_name == "seat_heating":
#             if "seat_heating" in yelp_poi and seat_position:
#                 key = seat_position_map.get(seat_position)
#                 if key and key in yelp_poi["seat_heating"]:
#                     setattr(output, system_field, "seat_heating")
#                     setattr(output, seat_position_field, seat_position)

#                     state = yelp_poi["seat_heating"][key]  # "off", "1", etc.
#                     if state == "off":
#                         level = 0
#                     elif state == "low":
#                         level = 1
#                     elif state == "medium":
#                         level = 2
#                     elif state == "high":
#                         level = 3
#                     elif isinstance(state, int) or isinstance(state, float):
#                         level = int(state)
#                     else:
#                         level = 0
#                     setattr(output, target_seat_heat_field, level)

#         return output

#     # -------------------------
#     # Procesar system 1
#     # -------------------------
#     if content_input.system:
#         fill_system(
#             system_field="system",
#             position_field="position",
#             seat_position_field="seat_position",
#             fog_pos_field="fog_light_position",
#             target_window_field="window_state_target",
#             target_onoff_field="onoff_state_target",
#             target_headlights_field="head_lights_mode_target",
#             target_temp_field="climate_temperature_value_target",
#             target_seat_heat_field="seat_heating_level_target",
#             system_name=content_input.system,
#             position=content_input.position,
#             seat_position=content_input.seat_position,
#         )

#     # -------------------------
#     # Procesar system 2
#     # -------------------------
#     if content_input.system2:
#         fill_system(
#             system_field="system2",
#             position_field="position2",
#             seat_position_field="seat_position2",
#             fog_pos_field="fog_light_position2",
#             target_window_field="window_state_target2",
#             target_onoff_field="onoff_state_target2",
#             target_headlights_field="head_lights_mode_target2",
#             target_temp_field="climate_temperature_value_target2",
#             target_seat_heat_field="seat_heating_level_target2",
#             system_name=content_input.system2,
#             position=content_input.position2,
#             seat_position=content_input.seat_position2,
#         )
#     print("output:", output)
#     input()
#     return output

