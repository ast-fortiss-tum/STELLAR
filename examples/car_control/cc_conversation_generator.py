import random
from typing import List, Optional

from llm.model.models import Conversation
#from examples.car_control.models import NaviContentInput, StyleDescription, NaviUserIntent
from examples.car_control.models_new import CCContentInput, StyleDescription, CCUserIntent
from llm.model.conversation_intents import OPTIMIZABLE_USER_INTENTS
from llm.features import FeatureHandler
from llm.conversation_generation.conversation_generator import ConversationGenerator
from llm.utils.seed import set_seed
#from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from examples.car_control.cc_utterance_generator_new import CCUtteranceGenerator


class CCConversationGenerator(ConversationGenerator):
    def __init__(
        self,
        feature_handler: Optional[FeatureHandler] = None,
        apply_constraints_to_vars: bool = True,
    ):
        super().__init__(feature_handler=feature_handler)
        self.apply_constraints_to_vars = apply_constraints_to_vars
        # Reuse the utterance generator's constraint logic
        self._utterance_gen = CCUtteranceGenerator(feature_handler=feature_handler)

    def apply_constraints(self, content_input: CCContentInput) -> CCContentInput:
        """Apply category-based constraints to content input.
        Delegates to the utterance generator to ensure alignment."""
        return self._utterance_gen.apply_constraints(content_input)

    def apply_constraints_style(self, content_input: CCContentInput, style_input: StyleDescription) -> StyleDescription:
        """Apply style constraints. Delegates to the utterance generator to ensure alignment."""
        return self._utterance_gen.apply_constraints_style(content_input, style_input)

    def _update_vars_from_content_input(
        self,
        categorical_vars: List[int],
        content_input: CCContentInput,
    ) -> List[int]:
        """Update categorical vars based on constrained content input."""
        updated_vars = list(categorical_vars)
        
        for i, (feature_name, feature) in enumerate(self.feature_handler.categorical_features.items()):
            if i >= len(updated_vars):
                break
            
                
            content_value = getattr(content_input, feature_name, None)
            
            # if constraint removed this value, update the var
            if content_value is None and categorical_vars[i] is not None:
                updated_vars[i] = 0
            elif content_value is not None:
                # update to match the constrained value
                var = self.feature_handler.get_var_from_feature_value(
                    feature=feature,
                    value=content_value,
                    feature_type=feature.feature_type
                )
                if var is not None:
                    updated_vars[i] = var
        
        return updated_vars

    def generate_conversation(
        self,
        ordinal_vars: List[float],
        categorical_vars: List[int],
        continuous_vars: List[float],
    ) -> Conversation:
        """
        Generate a half-empty conversation with empty turns -> no llm calling.
        """
        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
        )
        
        content_input = CCContentInput.model_validate(feature_values)
        content_input = self.apply_constraints(content_input)

        if self.apply_constraints_to_vars:
            categorical_vars = self._update_vars_from_content_input(
                categorical_vars,
                content_input
            )

        style_input = StyleDescription.model_validate(feature_values)
        style_input = self.apply_constraints_style(content_input, style_input)

        # TODO: synchronise NaviUserIntet, UserIntent & OptimizableUserIntent
        # validate intents
        intent_values = {}
        if continuous_vars:
            for i, intent_name in enumerate(OPTIMIZABLE_USER_INTENTS):
                if i < len(continuous_vars):
                    # NOTE: mutation operator might produce 0.0 if problem.xl is 0
                    val = continuous_vars[i]
                    if val < 0.001:
                        val = 0.001
                    elif val > 1.0:
                        val = 1.0
                    
                    continuous_vars[i] = val
                    intent_values[intent_name] = val
            CCUserIntent.model_validate(intent_values)

        conversation = Conversation(
            ordinal_vars=list(ordinal_vars),
            categorical_vars=list(categorical_vars),
            continuous_vars=list(continuous_vars),
            style_input=style_input,
            content_input_values=content_input.model_dump(exclude_none=True),
        )
        
        return conversation


if __name__ == "__main__":
    fhandler = FeatureHandler.from_json("configs/features_simple_judge_cc.json")
    set_seed(100)
    
    gen = CCConversationGenerator(fhandler)
    
    sampled = fhandler.sample_feature_scores()
    ordinal_vars = sampled.ordinal
    categorical_vars = sampled.categorical
    continuous_vars = sampled.continuous

    conv = gen.generate_conversation(
        ordinal_vars=ordinal_vars,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )
    
    print(f"Num turns: {len(conv)}")
    print(f"Ordinal vars: {conv.ordinal_vars}")
    print(f"Categorical vars: {conv.categorical_vars}")
    print(f"Continuous vars (intent priorities): {conv.continuous_vars}")
    print(f"Style input: {conv.style_input}")
    print(f"Content input values: {conv.content_input_values}")