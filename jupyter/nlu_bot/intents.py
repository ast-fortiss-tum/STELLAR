"""Intent catalog used by the NLU chatbot."""

INTENTS = [
    "INTENT_ActivateAirConditioning",
    "INTENT_ActivateClimateSync",
    "INTENT_ActivateFan",
    "INTENT_ActivateSeatCooling",
    "INTENT_ActivateSeatHeating",
    "INTENT_ActivateSteeringWheelHeating",
    "INTENT_DeactivateAirConditioning",
    "INTENT_DeactivateClimateSync",
    "INTENT_DeactivateFan",
    "INTENT_DeactivateSeatCooling",
    "INTENT_DeactivateSeatHeating",
    "INTENT_DeactivateSteeringWheelHeating",
    "INTENT_DecreaseFanSpeed",
    "INTENT_DecreaseSeatCooling",
    "INTENT_DecreaseSeatHeating",
    "INTENT_DecreaseTemperature",
    "INTENT_IncreaseFanSpeed",
    "INTENT_IncreaseSeatCooling",
    "INTENT_IncreaseSeatHeating",
    "INTENT_IncreaseTemperature",
    "INTENT_SetFanSpeed",
    "INTENT_SetSeatCooling",
    "INTENT_SetSeatHeating",
    "INTENT_SetTemperature",
    "INTENT_Unknown"
]

INTENT_SET = set(INTENTS)

__all__ = ["INTENTS", "INTENT_SET"]