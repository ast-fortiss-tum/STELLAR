from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.llms import LLMType, pass_llm
from jupyter.nlu_bot.intents import INTENTS
from jupyter.nlu_bot.ipa_nlu_bot import IPA_NLU_BOT, FALLBACK_INTENT


def _build_system_prompt() -> str:
    intents_text = IPA_NLU_BOT._build_indexed_intents_text()
    few_shot_json = IPA_NLU_BOT._build_few_shot_example_json()
    return (
        "Role: You are an intent classifier for in-car assistant utterances.\n\n"
        "Task:\n"
        "For the given utterance, provide confidence scores for every allowed intent listed below.\n\n"
        "Allowed intents (index: name):\n"
        f"{intents_text}\n\n"
        "Output format (strict):\n"
        '1) Return exactly one JSON object.\n'
        '2) The JSON object must contain exactly one key: "intent_confidences".\n'
        '3) "intent_confidences" must be a list with exactly one object per allowed intent (no missing, no extra).\n'
        '4) Each list object must contain exactly these keys: "intent_index", "confidence".\n'
        f'5) "intent_index" must be an integer in [0, {len(INTENTS) - 1}].\n'
        '6) "intent_index" values must be unique.\n'
        '7) "confidence" must be a number in [0.0, 1.0].\n'
        '8) Confidence values are independent and do not need to sum to 1.0.\n'
        '9) Do not output markdown, code fences, explanations, or any extra text.\n\n'
        '10) INTENT_Unknown is a special intent that should be used when the utterance does not match any of the other intents.\n\n'
        "Few-shot example:\n"
        'Utterance: "increase the fan speed please"\n'
        f"Output: {few_shot_json}"
    )


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _render_sidebar() -> tuple[LLMType, float, int]:
    st.sidebar.header("Bot Settings")

    model_names = [model.name for model in LLMType]
    default_model_index = model_names.index("GPT_4O") if "GPT_4O" in model_names else 0

    model_name = st.sidebar.selectbox("Model", model_names, index=default_model_index)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=16, max_value=2024, value=2000, step=16)

    if st.sidebar.button("Clear chat history"):
        st.session_state["messages"] = []
        st.rerun()

    return LLMType[model_name], float(temperature), int(max_tokens)


def _render_prediction(
    intent: str,
    certainty: float,
    top_intents: List[Dict[str, float]],
    intent_confidences: List[Dict[str, float]],
) -> None:
    st.markdown(
        (
            "<div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.35rem;'>"
            f"Top intent: {intent}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.progress(max(0.0, min(1.0, certainty)))
    st.caption(f"Top confidence: {certainty:.0%}")

    st.markdown("**Top 3 intents**")
    for rank, candidate in enumerate(top_intents[:3], start=1):
        item_intent = str(candidate.get("intent", FALLBACK_INTENT))
        item_confidence = float(candidate.get("confidence", 0.0))
        confidence_label = f"{item_confidence * 100:.1f}".replace(".", ",") + "%"
        st.write(f"{rank}. {item_intent} ({confidence_label})")

    sorted_all = sorted(
        intent_confidences,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    table_rows = [
        {
            "intent": str(item.get("intent", FALLBACK_INTENT)),
            "confidence": f"{float(item.get('confidence', 0.0)):.2%}",
        }
        for item in sorted_all
    ]
    with st.expander("All intent probabilities"):
        st.dataframe(table_rows, use_container_width=True, height=360)


def _render_history(messages: List[Dict[str, Any]]) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "intent" in msg and "certainty" in msg:
                intent_confidences = msg.get("intent_confidences") or IPA_NLU_BOT._fallback_intent_confidences(
                    str(msg["intent"]),
                    float(msg["certainty"]),
                )
                top_intents = IPA_NLU_BOT._top_k_from_intent_confidences(intent_confidences)
                _render_prediction(
                    str(msg["intent"]),
                    float(msg["certainty"]),
                    top_intents,
                    intent_confidences,
                )
            else:
                st.markdown(str(msg["content"]))


def _parse_prediction_from_llm_output(
    raw_output: str,
) -> Tuple[str, float, List[Dict[str, float]], List[Dict[str, float]]]:
    payload_text = raw_output.strip()
    print(f"LLM output: {payload_text}")
    intent, certainty, top_intents, intent_confidences = IPA_NLU_BOT._parse_prediction(payload_text)
    return intent, certainty, top_intents, intent_confidences


def _format_bot_response(intent: str, certainty: float, top_intents: List[Dict[str, float]]) -> str:
    lines = [f"TOP INTENT: {intent}", f"TOP CONFIDENCE: {certainty:.2f}", "TOP 3:"]
    for rank, candidate in enumerate(top_intents[:3], start=1):
        lines.append(
            f"{rank}. {candidate.get('intent', FALLBACK_INTENT)} ({float(candidate.get('confidence', 0.0)):.2f})"
        )
    return "\n".join(lines)


def _query_bot(
    user_message: str,
    model: LLMType,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, float, List[Dict[str, float]], List[Dict[str, float]]]:
    max_attempts = 3
    for attempt in range(max_attempts):
        llm_output = pass_llm(
            msg=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            llm_type=model,
            system_message=_build_system_prompt(),
        )
        print(f"LLM raw output attempt {attempt + 1}/{max_attempts}: {llm_output}")
        raw_intent_confidences = IPA_NLU_BOT._extract_intent_confidences(llm_output)
        # Fill missing intents with 0.0 confidence so partial model outputs remain usable.
        normalized_intent_confidences = IPA_NLU_BOT._normalize_intent_confidences(llm_output)
        is_valid, invalid_reason = IPA_NLU_BOT._validate_intent_confidences(normalized_intent_confidences)
        if is_valid:
            if invalid_reason != "ok":
                print(f"Validation note: {invalid_reason}")
            if len(raw_intent_confidences) != len(INTENTS):
                print(
                    f"Validation note: response had {len(raw_intent_confidences)}/{len(INTENTS)} intents; "
                    "missing intents were filled with confidence 0.0"
                )
            intent, certainty, top_intents, intent_confidences = _parse_prediction_from_llm_output(llm_output)
            return intent, certainty, top_intents, intent_confidences

        print(
            f"Invalid LLM top-intents output on attempt {attempt + 1}/{max_attempts}: {invalid_reason}. "
            f"Raw output: {llm_output}"
        )

    raise ValueError("Failed to obtain valid full intent probabilities from LLM after retries")


def main() -> None:
    st.set_page_config(page_title="In-Car Climate Assistant", page_icon="R", layout="centered")
    st.title("In-Car Climate Assistant")
    st.caption("Provide your utterance to activate climate functions.")

    _init_state()
    model, temperature, max_tokens = _render_sidebar()

    _render_history(st.session_state.messages)

    user_message = st.chat_input("Type your message...")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    intent, certainty, top_intents, intent_confidences = _query_bot(
                        user_message=user_message,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception:
                    intent_confidences = IPA_NLU_BOT._uniform_intent_confidences()
                    fallback_top_intents = IPA_NLU_BOT._top_k_from_intent_confidences(intent_confidences)

                    intent, certainty, top_intents = (
                        fallback_top_intents[0]["intent"],
                        fallback_top_intents[0]["confidence"],
                        fallback_top_intents,
                    )

            _render_prediction(intent, certainty, top_intents, intent_confidences)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": _format_bot_response(intent, certainty, top_intents),
                "intent": intent,
                "certainty": certainty,
                "top_intents": top_intents,
                "intent_confidences": intent_confidences,
            }
        )


if __name__ == "__main__":
    main()
