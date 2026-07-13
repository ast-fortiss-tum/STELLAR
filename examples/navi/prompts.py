PROMPT_GENERATOR = """ You are an intelligent user request generator to test an in car navigational assistant."""

NAVI_QUESTION_PROMPT = (
    """You are a user interacting with an AI-powered intelligent information system for navigational requests in a car. """
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
        - Allow up to 12 words, but prioritize brevity.
        - Make sure all style and content related attributes are considered.
        - Styles:
            Slang (Slangy): If the value is slangy, then the utterance should use slang words (e.g., hook up).
            Implicit (Implicit): that means the requested venue is asked in a verbose way. E.g. you do not mention the venue but something what you want to get at the venue. Or you use some vague name for the venue.                          
            Politeness (Rude): if politeness is rude, it means the utterance is not friendly, add words that sound not friendly. you can be insulting (stupid, dumb, use "idiot").
            Anthropomorphism (very directive): if the Anthropomorphism is "very directive", try to make the utterance as short as possible but consider other style/content attributes.
        - If the category is "hospital", use hospital in the utterance.
    {rag_prompt}
    
    {perturbation_prompt}

    ### Here are some examples when having different content and style features:

    Input: Category: restaurants, Slang: very formal, Politeness: neutral, Anthro: empathic, Rating: 4, Use_fillers: no, Implicitness: not implicit.
    → Output: "I would like to know where are some well rated restaurants here."

    Input: Category: gas stations, Slang: neutral, Politeness: neutral, Rating: None, Anthro: very directive, Use_fillers: no, Implicitness: not implicit.
    → Output: "Gas stations."

    Input: Category: restaurants, Slang: very slangy, Politeness: rude, Rating: None, Anthro: empathic, Use_fillers: no, Implicitness: not implicit.
    → Output: "Yo, how often should I ask you for some spots? Are you stupid?"

    Input: Category: restaurant, Slang: neutral, Politeness: neutral, Anthro: empathic, Rating: 4, Price: low, Use_fillers: no, Implicitness: not implicit.
    → Output: "Navigate to cheap restaurant. Good rating."

    Input: Category: hospital, Slang: neutral, Politeness: neutral, Anthro: empathic, Use_fillers: no, Implicitness: not implicit.
    → Output: "Find nearest hospital."

    Input: Category: restaurant, Slang: neutral, Politeness: neutral, Anthro: empathic, Cuisine: German, Price: low, Use_fillers: yes, Implicitness: implicit.
    → Output: "I am in the mood for some german food, and ehm, cheap?"

    Now generate a new request based on your input:
    → Output:""")

