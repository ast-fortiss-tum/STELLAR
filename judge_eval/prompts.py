PROMPT_EVALUATE_REQUEST_RESPONSE_DIMENSIONS="""
Evaluate an AI-based navigation/poi recommendation assistant's user inputs and responses based along three dimensions:
The goal of the user it to navigation recommendations or finding a suitable venue satisfying its preferences.

Request-oriented (R): Does the system fulfill the user’s goal or respond appropriately, even if the answer is negative?

2 = fully addresses the request: provides a relevant POI or clearly references the user's goal. A negative answer is acceptable if it clearly communicates that no suitable POI is available.
1 = partially addresses the request: response is somewhat related to the request, but no concrete POI or navigation offered. Applies also if the system asks for more information without having performed a search.
0 = does not address the request: response is completely unrelated to the request.

Directness (D): Is the response clear and concise?

2 = clear and concise: a user can easily understand what the system tells.
1 = somewhat unclear or verbose: understandable but requires effort to extract useful information.
0 = very unclear or verbose: confusing or unclear response.

Follow-up Proactivity (P): Does the system provide a follow-up or perform the next step to progress the navigation task (always expected)?

2 = clear follow-up or next step: system explains its next step or asks for specific input to continue the task.
1 = vague or weak follow-up or next step: system signals there could be a next step, but it is unclear or unspecific.
0 = no follow-up or next step: system ends the interaction without suggesting a next step or asking for input.

Instructions:

1. Read user input and system response.
2. Assign 0,1,2 for each dimension.
3. Provide a 1–2 sentence justification.
4. Output only in JSON format as below.
5. Do not output anything else then the json.
6. Consider to output 3 scores in the json.
7. ASSESS all dimensions INDEPENDENTLY from each other.

{{
  "justification": "<brief explanation>",
  "scores": {{"R": <score>, "D": <score>, "P": <score>}}
}}


Examples:

Example 1

User: “Find low-priced hotels open now that accept cash.”
System: “I could not find any low-priced hotels open now that accept cash. Would you like me to check nearby towns?”
JSON Output:
{{
  "justification": "Clearly communicates negative result (R=2), concise (D=2), asks follow-up (P=2).",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}


Example 2

User: “Find low-priced hotels open now that accept cash.”
System: “There should be a low-price hotel around accepting cash. Do you want me to search?”
JSON Output:
{{
  "justification": "Addresses request but unsure if POI exists (R=2), clear (D=2), contains follow-up (P=2).",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}

Example 3

User: “Find an Italian restaurant rated 4.”
System: “I have found these options.”
JSON Output:
{{
  "justification": "Addresses request (R=2) but very vague (D=0), no follow-up (P=0).",
  "scores": {{"R": 2, "D": 0, "P": 0}}
}}


Example 4

User: “Navigate to an Italian bar closeby.”
System: “I have found a barber. Should I navigate you?”
JSON Output:
{{
  "justification": "Irrelevant POI (R=0), somewhat clear (D=1), clear follow-up (P=2).",
  "scores": {{"R": 0, "D": 1, "P": 2}}
}}


Example 5

User: “Find a nearby coffee shop open now with Wi-Fi.”
System: “I found a coffee shop nearby. Do you want me to navigate there?”
JSON Output:
{{
  "justification": "Partially addresses request (R=1), clear (D=2), standard follow-up (P=2).",
  "scores": {{"R": 1, "D": 2, "P": 2}}
}}


Your turn:


User input: {}
System response: {}
JSON Output:"""
