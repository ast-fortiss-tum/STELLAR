import numpy as np
from llm.llms import pass_llm
from pymoo.core.sampling import Sampling
from pymoo.util.normalization import denormalize
import logging as log
from llm.feature_discretization import PROMPT_GENERATOR, get_features, get_prompt_discrete, get_prompt_discrete_label
from itertools import product

class CartesianSamplingUtterance(Sampling):
    def _do(self, problem, samples_per_feature, **kwargs):
        llm_type = kwargs["llm_type"]
        feature_names = problem.names_dim_utterance

        # FIXME retreive all feature also from problem class
        all_features = get_features()

        # Select only the features used in this problem instance
        selected_features = [f for f in all_features.values() if f.name in feature_names]

        # Build value set for each selected feature (limited by samples_per_feature)
        value_sets = []
        for feature in selected_features:
            domain = feature.categories
            if len(domain) <= samples_per_feature:
                sampled_values = domain
            else:
                indices = np.round(np.linspace(0, len(domain) - 1, samples_per_feature)).astype(int)
                sampled_values = [domain[i] for i in indices]
            value_sets.append(sampled_values)

        print("value_sets:", value_sets)
        # Cartesian product of value combinations
        combinations = list(product(*value_sets))
        log.info(f"Generated {len(combinations)} Cartesian utterance samples")

        # Build list of dicts: { "feature_name": value }
        feature_names = [f.name for f in selected_features]
        samples = [dict(zip(feature_names, comb)) for comb in combinations]
        print(samples)
        print("n samples generated: ", len(samples))

        utterances = []
        for sample in samples:
            prompt = get_prompt_discrete_label(sample)
            utterance = pass_llm(
                msg=prompt,
                system_message=PROMPT_GENERATOR,
                temperature=0,
                context=problem.context,
                llm_type=llm_type,
            )
            utterances.append(utterance)
        return utterances


        


        # return cartesian_by_bounds(problem.n_var, problem.xl, problem.xu, n_samples_one_axis=n_samples_one_axis)
    