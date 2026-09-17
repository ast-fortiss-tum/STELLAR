import random
from typing import List, Tuple, Dict, Set, Optional, Any

from llm.sut.ipa_base import IPABase
from llm.features import FeatureHandler
from llm.features.models import CombinedFeaturesInstance, FeatureType
from llm.prompts import CONVERSATION_FOLLOW_UP_PROMPTS_NAVI
from llm.model.models import ContentInput

from examples.navi.models import NaviContentInput

PRICE_FEATURE_NAME = "fuel_price"


class IPABaseNavi(IPABase):
    FOLLOW_UP_PROMPTS = CONVERSATION_FOLLOW_UP_PROMPTS_NAVI
    ANCHOR_KEYS = ["category"]
    content_input_class = NaviContentInput

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
        prev_price = prev_dump.get(PRICE_FEATURE_NAME) if prev_dump else None

        cat_keys = list(feature_handler.categorical_features.keys())
        ord_keys = list(feature_handler.ordinal_features.keys())

        for _ in range(max_attempts):
            scores: CombinedFeaturesInstance = feature_handler.sample_feature_scores()

            new_cat = list(scores.categorical)

            preserved_cat_features = ["category"]
            preserved_cat_features.extend(
                key for key in cat_keys if "perturbation" in key
            )

            for feature_name in preserved_cat_features:
                if feature_name in cat_keys and original_categorical_vars:
                    idx = cat_keys.index(feature_name)
                    if idx < len(new_cat) and idx < len(original_categorical_vars):
                        new_cat[idx] = original_categorical_vars[idx]

            new_ord = list(original_ordinal_vars)

            if "rating" in ord_keys:
                idx = ord_keys.index("rating")
                if idx < len(new_ord) and idx < len(scores.ordinal):
                    new_ord[idx] = scores.ordinal[idx]

            if prev_price is not None:
                if PRICE_FEATURE_NAME in ord_keys:
                    price_idx = ord_keys.index(PRICE_FEATURE_NAME)
                    if price_idx < len(new_ord):
                        lower_price_var = _pick_lower_price_var(prev_price, feature_handler)
                        if lower_price_var is not None:
                            new_ord[price_idx] = lower_price_var
                        elif price_idx < len(original_ordinal_vars):
                            new_ord[price_idx] = original_ordinal_vars[price_idx]
                elif PRICE_FEATURE_NAME in cat_keys:
                    price_idx = cat_keys.index(PRICE_FEATURE_NAME)
                    if price_idx < len(new_cat):
                        feat = feature_handler.categorical_features[PRICE_FEATURE_NAME]
                        new_price_val = feat.values[new_cat[price_idx]]
                        if new_price_val is None or new_price_val > prev_price:
                            valid_indices = [
                                i for i, v in enumerate(feat.values)
                                if v is not None and v < prev_price
                            ]
                            if valid_indices:
                                new_cat[price_idx] = random.choice(valid_indices)
                            elif price_idx < len(original_categorical_vars):
                                new_cat[price_idx] = original_categorical_vars[price_idx]

            if prev_dump is None:
                return new_cat, new_ord

            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord,
                categorical_feature_indices=new_cat,
            )
            has_difference = False
            for k, v in new_vals.items():
                if k == "category":
                    continue
                if prev_dump.get(k) != v:
                    has_difference = True
                    break

            if has_difference:
                return new_cat, new_ord

        new_vals = feature_handler.get_feature_values_dict(
            ordinal_feature_scores=new_ord,
            categorical_feature_indices=new_cat,
        )
        mutable_keys = [
            k for k in new_vals
            if k != "category" and "perturbation" not in k
        ]

        if mutable_keys:
            force_key = random.choice(mutable_keys)
            if force_key in cat_keys:
                idx = cat_keys.index(force_key)
                feat = feature_handler.categorical_features[force_key]
                possible_indices = list(range(len(feat.values)))
                current_idx = new_cat[idx]
                other_indices = [i for i in possible_indices if i != current_idx]
                if other_indices:
                    new_cat[idx] = random.choice(other_indices)
            elif force_key in ord_keys:
                idx = ord_keys.index(force_key)
                if force_key == PRICE_FEATURE_NAME and prev_price is not None:
                    lower_price_var = _pick_lower_price_var(prev_price, feature_handler)
                    if lower_price_var is not None:
                        new_ord[idx] = lower_price_var
                else:
                    current_val = new_ord[idx]
                    for _ in range(50):
                        candidate = random.random()
                        if candidate != current_val:
                            new_ord[idx] = candidate
                            break

        return new_cat, new_ord

    @staticmethod
    def initialize_content_features(
        initial_content_input: ContentInput,
        sample: bool = True,
    ) -> Tuple[Set[str], Set[str], ContentInput]:
        all_content_features = {
            k
            for k, v in initial_content_input.model_dump().items()
            if v is not None and k != "category"
        }

        content_input_turn1 = initial_content_input.model_copy()

        valid_fields = list(all_content_features)

        if valid_fields:
            if sample:
                features_to_keep = [random.choice(valid_fields)]
            else:
                features_to_keep = valid_fields

            features_to_remove = [f for f in valid_fields if f not in features_to_keep]

            for field_name in features_to_remove:
                setattr(content_input_turn1, field_name, None)

            used_content_features = set(features_to_keep)
        else:
            used_content_features = set()

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
        prev_price = prev_dump.get(PRICE_FEATURE_NAME)

        changed_features: Dict[str, Any] = {}
        current_content_input = previous_content_input
        new_vals = {}

        for attempt in range(max_resample_attempts):
            new_cat_vars, new_ord_vars = IPABaseNavi.resample_content(
                feature_handler,
                categorical_vars,
                ordinal_vars,
                previous_content_input=previous_content_input,
            )

            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord_vars,
                categorical_feature_indices=new_cat_vars,
            )

            current_content_input = NaviContentInput.model_validate(new_vals)
            current_content_input = utterance_gen.apply_constraints(current_content_input)

            new_price = getattr(current_content_input, PRICE_FEATURE_NAME, None)
            if prev_price is not None:
                if new_price is None:
                    setattr(current_content_input, PRICE_FEATURE_NAME, prev_price)
                elif new_price >= prev_price:
                    lower = _pick_lower_price(prev_price, feature_handler)
                    if lower is not None:
                        setattr(current_content_input, PRICE_FEATURE_NAME, lower)
                    else:
                        setattr(current_content_input, PRICE_FEATURE_NAME, prev_price)

            new_dump = current_content_input.model_dump(exclude_none=True)

            changed_features = {}
            for k, v in new_dump.items():
                if k == "category":
                    continue
                prev_v = prev_dump.get(k)
                if prev_v != v:
                    changed_features[k] = v

            for k in prev_dump:
                if k == "category":
                    continue
                if k not in new_dump:
                    changed_features[k] = None

            non_changed_keys = [
                k for k, v in new_dump.items()
                if (
                    k != "category"
                    and k not in changed_features
                    and v is not None
                    and not (k == PRICE_FEATURE_NAME and prev_price is not None)
                )
            ]
            if non_changed_keys:
                num_to_omit = random.randint(0, len(non_changed_keys))
                keys_to_omit = random.sample(non_changed_keys, num_to_omit)
                for k in keys_to_omit:
                    setattr(current_content_input, k, None)
                    changed_features[k] = None

            concrete_changed = {k: v for k, v in changed_features.items() if v is not None}

            if concrete_changed:
                break

            forceable = [
                k for k, v in new_dump.items()
                if k != "category" and v is not None
            ]

            if forceable:
                force_key = random.choice(forceable)
                changed_features[force_key] = new_dump[force_key]
                break

        concrete_changed = {k: v for k, v in changed_features.items() if v is not None}
        if not concrete_changed:
            new_dump_final = current_content_input.model_dump(exclude_none=True)
            for k, v in new_dump_final.items():
                if k != "category":
                    changed_features[k] = v

        all_content_features = {
            k
            for k, v in current_content_input.model_dump().items()
            if v is not None and k != "category"
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
        unused = all_content_features - used_content_features
        if unused:
            prev_price = getattr(current_content_input, PRICE_FEATURE_NAME, None)
            filtered_unused = set(unused)

            if PRICE_FEATURE_NAME in filtered_unused and prev_price is not None:
                if _is_lowest_price(prev_price, feature_handler):
                    filtered_unused.discard(PRICE_FEATURE_NAME)

            if filtered_unused:
                feature_to_add = random.choice(list(filtered_unused))

                if feature_to_add == PRICE_FEATURE_NAME and prev_price is not None:
                    lower = _pick_lower_price(prev_price, feature_handler)
                    if lower is not None:
                        setattr(current_content_input, PRICE_FEATURE_NAME, lower)
                    else:
                        filtered_unused.discard(PRICE_FEATURE_NAME)
                        if filtered_unused:
                            feature_to_add = random.choice(list(filtered_unused))
                        else:
                            feature_to_add = None

                if feature_to_add is not None:
                    used_content_features.add(feature_to_add)
                    new_feature_this_turn = feature_to_add

        return current_content_input, used_content_features, new_feature_this_turn

    @staticmethod
    def _build_fallback_user_utterance(
        current_content_input: Optional[ContentInput],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = ["Find me"]
        feature_parts = []

        if current_content_input is not None:
            d = current_content_input.model_dump(exclude_none=True)

            if new_feature_this_turn and new_feature_this_turn in d:
                feature_parts.append(f"{new_feature_this_turn}: {d[new_feature_this_turn]}")
            elif changed_features:
                for k, v in changed_features.items():
                    if v is not None:
                        feature_parts.append(f"{k}: {v}")
            else:
                if "category" in d:
                    feature_parts.append(str(d["category"]).replace("_", " "))

        if feature_parts:
            parts.append("a place with " + ", ".join(feature_parts))
        else:
            parts.append("a suitable place nearby")

        return " ".join(parts)

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
        filtered: Dict[str, Any] = {}

        filtered["category"] = full_dump.get("category")

        for feat in used_content_features:
            if feat in full_dump and full_dump[feat] is not None:
                filtered[feat] = full_dump[feat]

        if new_feature_this_turn and new_feature_this_turn in full_dump:
            if full_dump[new_feature_this_turn] is not None:
                filtered[new_feature_this_turn] = full_dump[new_feature_this_turn]

        return NaviContentInput.model_validate(filtered)


# --- Price helper functions ---

def _get_ordered_price_values(feature_handler: FeatureHandler) -> List[float]:
    feat = feature_handler.categorical_features.get(PRICE_FEATURE_NAME)
    if feat is None:
        feat = feature_handler.ordinal_features.get(PRICE_FEATURE_NAME)
    if feat is None:
        return []
    return sorted(value for value in feat.values if value is not None)


def _is_lowest_price(current_price, feature_handler: FeatureHandler) -> bool:
    ordered = _get_ordered_price_values(feature_handler)
    if not ordered or current_price is None:
        return False
    return current_price <= ordered[0]


def _pick_lower_price(current_price, feature_handler: FeatureHandler):
    ordered = _get_ordered_price_values(feature_handler)
    if not ordered or current_price is None:
        return None
    candidates = [p for p in ordered if p < current_price]
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_lower_price_var(current_price, feature_handler: FeatureHandler):
    feat = feature_handler.ordinal_features.get(PRICE_FEATURE_NAME)
    if feat is None or current_price is None:
        return None

    lower_price = _pick_lower_price(current_price, feature_handler)
    if lower_price is None:
        return None

    return feature_handler.get_var_from_feature_value(
        feature=feat,
        value=lower_price,
        feature_type=FeatureType.ORDINAL,
    )
