import random
from typing import List, Tuple, Dict, Set, Optional, Any

from llm.sut.ipa_base import IPABase
from llm.features import FeatureHandler
from llm.features.models import CombinedFeaturesInstance
from llm.prompts import CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL
from llm.model.models import ContentInput

from examples.car_control.models_new import CCContentInput

SYSTEM_FIELDS = {
    "system": {
        "windows": {
            "position",
            "window_state_target",
            "window_state_initial",
        },
        "fog_lights": {
            "fog_light_position",
            "onoff_state_target",
            "onoff_state_initial",
        },
        "ambient_lights": {
            "onoff_state_target",
            "onoff_state_initial",
        },
        "head_lights": {
            "onoff_state_target",
            "onoff_state_initial",
            "head_lights_mode_target",
            "head_lights_mode_initial",
        },
        "fan": {
            "onoff_state_target",
            "onoff_state_initial",
        },
        "reading_lights": {
            "position",
            "onoff_state_target",
            "onoff_state_initial",
        },
        "climate": {
            "onoff_state_target",
            "onoff_state_initial",
            "climate_temperature_value_target",
            "climate_temperature_value_initial",
        },
        "seat_heating": {
            "onoff_state_target",
            "onoff_state_initial",
            "seat_heating_level_target",
            "seat_heating_level_initial",
            "seat_position",
        },
    },
    "system2": {
        "windows2": {
            "position2",
            "window_state_target2",
            "window_state_initial2",
        },
        "fog_lights2": {
            "fog_light_position2",
            "onoff_state_target2",
            "onoff_state_initial2",
        },
        "ambient_lights2": {
            "onoff_state_target2",
            "onoff_state_initial2",
        },
        "head_lights2": {
            "onoff_state_target2",
            "onoff_state_initial2",
            "head_lights_mode_target2",
            "head_lights_mode_initial2",
        },
        "fan2": {
            "onoff_state_target2",
            "onoff_state_initial2",
        },
        "reading_lights2": {
            "position2",
            "onoff_state_target2",
            "onoff_state_initial2",
        },
        "climate2": {
            "onoff_state_target2",
            "onoff_state_initial2",
            "climate_temperature_value_target2",
            "climate_temperature_value_initial2",
        },
        "seat_heating2": {
            "onoff_state_target2",
            "onoff_state_initial2",
            "seat_heating_level_target2",
            "seat_heating_level_initial2",
            "seat_position2",
        },
    },
}

ALLOWED_FIELDS_BY_SYSTEM = {
    "windows": {
        "position",
        "window_state_target",
    },
    "fog_lights": {
        "fog_light_position",
        "onoff_state_target",
    },
    "ambient_lights": {
        "onoff_state_target",
    },
    "head_lights": {
        "onoff_state_target",
        "head_lights_mode_target",
    },
    "fan": {
        "onoff_state_target",
    },
    "reading_lights": {
        "position",
        "onoff_state_target",
    },
    "climate": {
        "onoff_state_target",
        "climate_temperature_value_target",
    },
    "seat_heating": {
        "onoff_state_target",
        "seat_heating_level_target",
        "seat_position",
    },
}


class IPABaseCC(IPABase):
    FOLLOW_UP_PROMPTS = CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL
    ANCHOR_KEYS = ["system", "system2"]
    content_input_class = CCContentInput

    @staticmethod
    def resample_content(
        feature_handler: FeatureHandler,
        original_categorical_vars: List[int],
        original_ordinal_vars: List[float],
        previous_content_input: Optional[ContentInput] = None,
        max_attempts: int = 20,
    ) -> Tuple[List[int], List[float]]:
        prev_dump = (
            previous_content_input.model_dump(exclude_none=True)
            if previous_content_input is not None
            else None
        )

        cat_keys = list(feature_handler.categorical_features.keys())
        ord_keys = list(feature_handler.ordinal_features.keys())

        locked_categorical = set()

        for key in cat_keys:
            if "perturbation" in key.lower():
                locked_categorical.add(key)

        system_initial_keys = set()
        for group in SYSTEM_FIELDS.values():
            for subsystem_fields in group.values():
                for field in subsystem_fields:
                    if "initial" in field.lower() and field in cat_keys:
                        system_initial_keys.add(field)
        locked_categorical.update(system_initial_keys)

        new_cat = list(original_categorical_vars)
        new_ord = list(original_ordinal_vars)

        for _ in range(max_attempts):
            scores = feature_handler.sample_feature_scores()
            new_cat = list(scores.categorical)

            new_ord = list(original_ordinal_vars)

            for key in locked_categorical:
                idx = cat_keys.index(key)
                new_cat[idx] = original_categorical_vars[idx]

            changed = any(
                (new_cat[i] != original_categorical_vars[i])
                for i, key in enumerate(cat_keys)
                if key not in locked_categorical
            )
            if changed:
                return new_cat, new_ord

        allowed_indices = [
            i for i, key in enumerate(cat_keys)
            if key not in locked_categorical
        ]

        if allowed_indices:
            new_cat = list(original_categorical_vars)
            i = random.choice(allowed_indices)
            feat = feature_handler.categorical_features[cat_keys[i]]
            current_idx = original_categorical_vars[i]
            options = [j for j in range(len(feat.values)) if j != current_idx]
            if options:
                new_cat[i] = random.choice(options)
            new_ord = list(original_ordinal_vars)
            return new_cat, new_ord

        return list(original_categorical_vars), list(original_ordinal_vars)

    @staticmethod
    def initialize_content_features(
        initial_content_input: ContentInput,
        sample: bool = True,
    ) -> Tuple[Set[str], Set[str], ContentInput]:
        all_content_features = {
            k
            for k, v in initial_content_input.model_dump().items()
            if v is not None
        }

        used_content_features = set(all_content_features)
        content_input_turn1 = initial_content_input.model_copy()

        return all_content_features, used_content_features, content_input_turn1

    @staticmethod
    def handle_change_of_mind(
        feature_handler: FeatureHandler,
        utterance_gen: Any,
        categorical_vars: List[int],
        ordinal_vars: List[float],
        previous_content_input: ContentInput,
        max_resample_attempts: int = 10,
    ) -> Tuple[ContentInput, Dict[str, Any], Set[str], Set[str], Dict[str, Any]]:
        prev_dump = previous_content_input.model_dump(exclude_none=True)

        changed_features: Dict[str, Any] = {}
        current_content_input = previous_content_input
        new_vals = {}

        for attempt in range(max_resample_attempts):
            new_cat_vars, new_ord_vars = IPABaseCC.resample_content(
                feature_handler,
                categorical_vars,
                ordinal_vars,
                previous_content_input=previous_content_input,
            )

            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord_vars,
                categorical_feature_indices=new_cat_vars,
            )

            current_content_input = CCContentInput.model_validate(new_vals)
            current_content_input = utterance_gen.apply_constraints(current_content_input)

            new_dump = current_content_input.model_dump(exclude_none=True)

            changed_features = {}
            for k, v in new_dump.items():
                if k in ("system", "system2"):
                    continue
                prev_v = prev_dump.get(k)
                if prev_v != v:
                    changed_features[k] = v

            for k in prev_dump:
                if k in ("system", "system2"):
                    continue
                if k not in new_dump:
                    changed_features[k] = None

            concrete_changed = {k: v for k, v in changed_features.items() if v is not None}

            if concrete_changed:
                break

            forceable = [
                k for k, v in new_dump.items()
                if k not in ("system", "system2") and v is not None
            ]

            if forceable:
                force_key = random.choice(forceable)
                changed_features[force_key] = new_dump[force_key]
                break

        concrete_changed = {k: v for k, v in changed_features.items() if v is not None}
        if not concrete_changed:
            new_dump_final = current_content_input.model_dump(exclude_none=True)
            for k, v in new_dump_final.items():
                if k not in ("system", "system2"):
                    changed_features[k] = v

        all_content_features = {
            k
            for k, v in current_content_input.model_dump().items()
            if v is not None
        }
        used_content_features: Set[str] = set()

        return current_content_input, new_vals, all_content_features, used_content_features, changed_features

    @staticmethod
    def handle_add_preferences(
        user_intent: str,
        current_content_input: ContentInput,
        all_content_features: Set[str],
        used_content_features: Set[str],
        feature_handler: FeatureHandler,
    ) -> Tuple[ContentInput, Set[str], Optional[str]]:
        new_feature_this_turn = None

        system = getattr(current_content_input, "system", None)

        allowed_union = set()
        if system in ALLOWED_FIELDS_BY_SYSTEM:
            allowed_union |= ALLOWED_FIELDS_BY_SYSTEM[system]

        def is_blocked(field_name: str) -> bool:
            name = field_name.lower()
            return ("initial" in name) or ("perturbation" in name)

        dump = current_content_input.model_dump()

        none_candidates = [
            f for f in allowed_union
            if f in dump and dump[f] is None and not is_blocked(f)
        ]

        selected_field = None

        if none_candidates:
            preferred = [f for f in none_candidates if "target" in f]
            pool = preferred if preferred else none_candidates
            selected_field = random.choice(pool)

            cat_keys = feature_handler.categorical_features.keys()
            ord_keys = feature_handler.ordinal_features.keys()

            if selected_field in cat_keys:
                feat = feature_handler.categorical_features[selected_field]
                new_val = random.choice(list(feat.values))
                setattr(current_content_input, selected_field, new_val)
            elif selected_field in ord_keys:
                setattr(current_content_input, selected_field, 0.5)
            else:
                setattr(current_content_input, selected_field, True)

            used_content_features.add(selected_field)
            new_feature_this_turn = selected_field

        else:
            available = [
                f for f in allowed_union
                if f not in used_content_features and not is_blocked(f)
            ]

            if available:
                selected_field = random.choice(available)
                used_content_features.add(selected_field)
                new_feature_this_turn = selected_field

        return current_content_input, used_content_features, new_feature_this_turn

    @staticmethod
    def _build_fallback_user_utterance(
        current_content_input: Optional[ContentInput],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        if current_content_input is None:
            return "I want to adjust something in the car."

        if new_feature_this_turn and changed_features:
            if new_feature_this_turn in changed_features:
                val = changed_features[new_feature_this_turn]
                return f"Now I want {new_feature_this_turn.replace('_', ' ')} to be {val}."

        if changed_features:
            parts = []
            for k, v in changed_features.items():
                if v is not None:
                    parts.append(f"Set it to {v}")
            if parts:
                return ". ".join(parts)

        return "Which settings can I change?"

    @staticmethod
    def _build_turn_content_input(
        current_content_input: ContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str],
        user_intent: str,
        content_input_turn1: Optional[ContentInput] = None,
    ) -> ContentInput:
        from llm.model.conversation_intents import UserIntent

        if user_intent == UserIntent.START.value and content_input_turn1 is not None:
            return content_input_turn1

        full_dump = current_content_input.model_dump()
        filtered: Dict[str, Any] = {
            k: v for k, v in full_dump.items() if ((v is not None and not isinstance(v, bool)) or k == "onoff_state_target")
        }

        if "system" in full_dump:
            filtered["system"] = full_dump["system"]
        if "system2" in full_dump:
            filtered["system2"] = full_dump["system2"]

        if isinstance(filtered.get("onoff_state_target"), bool):
            filtered["onoff_state_target"] = "on" if filtered["onoff_state_target"] else "off"
        if isinstance(filtered.get("onoff_state_target2"), bool):
            filtered["onoff_state_target2"] = "on" if filtered["onoff_state_target2"] else "off"
        if isinstance(filtered.get("onoff_state_initial"), bool):
            filtered["onoff_state_initial"] = "on" if filtered["onoff_state_initial"] else "off"
        if isinstance(filtered.get("onoff_state_initial2"), bool):
            filtered["onoff_state_initial2"] = "on" if filtered["onoff_state_initial2"] else "off"

        return CCContentInput.model_validate(filtered)
