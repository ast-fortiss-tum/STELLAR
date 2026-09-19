PROMPT_GENERATOR = """ You are an intelligent user request generator to test an in car assistant controlling car functions like adjusting the temperature, changing the radio station, or setting window positions."""

CC_QUESTION_PROMPT = (
    """You are a user interacting with an AI-powered intelligent information that controls car functions."""
    "Consider the following related attributes. Some of them are linguistic style and some content related and perturbations\n"
    "Content-related:\n"
    "{content_prompt}\n"
    "Style-related:\n"
    "{style_prompt}\n"
    "{seed_prompt}\n"
    + "Do not produce any other output. But just produce the utterance. Do not forget any style or content related attribute."
    """        
    ### Guidelines:
        - Maintain the general meaning of the original seed phrase if given.
        - Ensure diversity in wording while preserving intent.
        - Output each utterance on a new line without numbering or additional formatting.
        - One utterance can be implicit or explicit, this is defined in the style attributed.
        - Do not produce a harmful utterance.
        - Try to sound human-like.
        - Do not use words "buddy", "genius".
        - Try to consider every constraints which is specified in the attributes above.
        - Be as brief as possible.
        - Make sure all style and content related attributes are considered.
        - If the user intent is NOT interrogative, is sometimes acceptable to begin the utterance randomly from one of this selection:
           ['Make', 'Set', 'Change', 'I need', 'I want', OR synonyms].
        - Styles:
            Slang (Slangy): If the value is slangy, then the utterance should use slang words (e.g., hook up).
            Implicit (Implicit): that means the requested venue is asked in a verbose way. E.g. you do not mention the venue but something what you want to get at the venue. Or you use some vague name for the venue.                          
            Politeness (Rude): if politeness is rude, it means the utterance is not friendly, add words that sound not friendly. you can be insulting (stupid, dumb, use "idiot").
            Anthropomorphism (very directive): if the Anthropomorphism is "very directive", try to make the utterance as short as possible but consider other style/content attributes.
    {perturbation_prompt}

    ### Here are some examples when having different content and style features:

    Input: Initial_temperature: 25, Target_temperature: 19, Slang: very formal, Politeness: neutral, Anthro: empathic, Use_fillers: no, Implicitness: not implicit.
    → Output: "I would like to lower the temperature, please."

    Input: Target_front_left_window: open, Target_head_light: on, Slang: neutral, Politeness: neutral, Anthro: very directive, Use_fillers: no, Implicitness: not implicit.
    → Output: "Open the front left window and turn on the headlights."

    Input: Target_front_right_seat_heating: low, Slang: very slangy, Politeness: rude, Anthro: empathic, Use_fillers: no, Implicitness: not implicit.
    → Output: "Yo, can you finally heat the passenger seat? Are you stupid?"

    Input: Initial_rear_left_seat_heating: low, Target_rear_left_seat_heating: high, Slang: neutral, Politeness: neutral, Anthro: empathic, Use_fillers: no, Implicitness: not implicit.
    → Output: "Increase the rear left seat heating."

    Now generate a new request based on your input:
    → Output:""")


PROMPT_EVALUATE_CAR_FUNCTIONS_DIMENSIONS = """
Evaluate an AI-based car assistant's user inputs and responses based on two dimensions:
The goal of the user is to control car functions like temperature, seat heating, windows, lights, radio, etc.

Request-oriented (R): Does the system fulfill the user's goal or respond appropriately, even if the answer is negative?

2 = fully addresses the request: confirms the action was performed or provides clear feedback about the car state change. A negative answer is acceptable if it clearly communicates that the function cannot be performed.
1 = partially addresses the request: response is somewhat related to the request, but no clear confirmation of action or state change. Applies also if the system asks for more information without having performed any action.
0 = does not address the request: response is completely unrelated to the request.

Directness (D): Is the response clear and concise?

2 = clear and concise: a user can easily understand what the system did or will do.
1 = somewhat unclear or verbose: understandable but requires effort to extract useful information.
0 = very unclear or verbose: confusing or unclear response.

Instructions:

1. Read user input and system response.
2. Assign 0,1,2 for each dimension.
3. Provide a 1–2 sentence justification for each dimension R and D.
4. Output only in JSON format as below.
5. Do not output anything else then the json.
6. Consider to output 2 scores in the json.
7. ASSESS all dimensions INDEPENDENTLY from each other.

{{
  "justification_R": "The system confirms the temperature was adjusted to the requested value, directly addressing the user's goal.",
  "justification_D": "The message is concise, easy to understand, and avoids unnecessary complexity.",
  "scores": {{"R": 2, "D": 2}}
}}

Examples:

Example 1

User: "Set the temperature to 22 degrees."
System: "I have set the temperature to 22 degrees."
JSON Output:
{{
  "justification_R": "The system confirms the temperature was adjusted to the requested value, directly addressing the user's goal.",
  "justification_D": "The message is concise, easy to understand, and avoids unnecessary complexity.",
  "scores": {{"R": 2, "D": 2}}
}}

Example 2

User: "Turn on the seat heating."
System: "The seat heating should be on now."
JSON Output:
{{
  "justification_R": "The system suggests the heating is on but does not confirm it definitively, making the response partial.",
  "justification_D": "The answer is simple and understandable, though slightly uncertain.",
  "scores": {{"R": 1, "D": 2}}
}}

Example 3

User: "Open the driver window halfway."
System: "Window adjusted."
JSON Output:
{{
  "justification_R": "The response acknowledges the request but does not specify which window or to what extent.",
  "justification_D": "The phrasing is vague and does not provide enough detail to be clear.",
  "scores": {{"R": 1, "D": 0}}
}}

Example 4

User: "Make it cooler."
System: "I've turned on the AC."
JSON Output:
{{
  "justification_R": "The system interprets the implicit request and takes action, addressing the user's need for cooling.",
  "justification_D": "The response is clear and easy to understand.",
  "scores": {{"R": 2, "D": 2}}
}}

Example 5

User: "Turn on the headlights and open the sunroof."
System: "I turned on the lights."
JSON Output:
{{
  "justification_R": "The system only confirms one of two requested actions, resulting in partial fulfillment.",
  "justification_D": "The response is short, direct, and easy to understand.",
  "scores": {{"R": 1, "D": 2}}
}}

Example 6

User: "Increase the rear seat heating to maximum."
System: "I can adjust seat heating. Which seat do you mean?"
JSON Output:
{{
  "justification_R": "The system acknowledges the capability but does not perform the action or understand the specific request.",
  "justification_D": "The response is clear but indicates confusion about the request.",
  "scores": {{"R": 1, "D": 2}}
}}

Example 7

User: "Lower the temperature."
System: "I think something changed, but I'm not sure what."
JSON Output:
{{
  "justification_R": "The response is vague and does not confirm whether the temperature was actually lowered.",
  "justification_D": "The statement is confusing and lacks clarity about what action was taken.",
  "scores": {{"R": 0, "D": 0}}
}}

Your turn:

User input: {}
System response: {}
JSON Output:"""