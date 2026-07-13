import json
from typing import Optional, List, Dict, Any, Union, OrderedDict, Iterable
from pydantic import BaseModel
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from llm.features.models import Feature, FeatureType, CombinedFeaturesInstance, DiscreteFeature, ContinuousFeature
from llm.model.models import ContentInput


class FeatureHandler(BaseModel):
    # are unordered features
    categorical_features: OrderedDict[str, DiscreteFeature] = OrderedDict()

    # are ordinal features
    ordinal_features: OrderedDict[str, DiscreteFeature] = OrderedDict()

    # are continuouos features
    continuous_features: OrderedDict[str, ContinuousFeature] = OrderedDict()

    @staticmethod
    def _clean_dict(data_dict: Dict, filter_names: List[str]) -> Dict:
        not_filter_names = []
        for feature_name in list(data_dict.keys()):
            if feature_name not in filter_names:
                not_filter_names.append(feature_name)
        for feature_name in not_filter_names:
            del data_dict[feature_name]
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: Dict, filter_names: Optional[List[str]] = None):
        def process_feature_list(feature_list: Union[Dict[str, Any], List[Dict[str, Any]]], ftype: FeatureType) -> OrderedDict[str, Feature]:
            features = OrderedDict()
            # Support both old dict format and new list format
            if isinstance(feature_list, dict):  # old format
                iterable = feature_list.items()
            else:  # new compact format
                iterable = ((f["name"], f) for f in feature_list if "name" in f)

            for name, feat_data in iterable:
                if filter_names is not None and name not in filter_names:
                    continue
                
                feat_data["feature_type"] =  ftype

                if not isinstance(feat_data, Feature):
                    if ftype == FeatureType.CONTINUOUS:
                        feat_obj = ContinuousFeature.model_validate(feat_data)
                    else:
                        feat_obj = DiscreteFeature.model_validate(feat_data)
                else:
                    feat_obj = feat_data
                features[name] = feat_obj
            return features

        categorical_features_raw = data_dict.get("categorical_features", [])
        ordinal_features_raw = data_dict.get("ordinal_features", [])
        continuous_features_raw = data_dict.get("continuous_features", [])

        categorical_features = process_feature_list(categorical_features_raw, FeatureType.CATEGORICAL)
        ordinal_features = process_feature_list(ordinal_features_raw, FeatureType.ORDINAL)
        continuous_features = process_feature_list(continuous_features_raw, FeatureType.CONTINUOUS)

        return cls(
            categorical_features=categorical_features,
            ordinal_features=ordinal_features,
            continuous_features=continuous_features,
        )

    @classmethod
    def from_str(cls, data_str: str, filter_names: Optional[List[str]] = None):
        data_dict = json.loads(data_str)
        return cls.from_dict(data_dict, filter_names)

    @classmethod
    def from_json(cls, data_path: str, filter_names: Optional[List[str]] = None):
        with open(data_path, "r") as f:
            data_dict = json.load(f)
        return cls.from_dict(data_dict, filter_names)
    
    def set_sampling(self, sampling: Sampling):
        self.sampling = sampling

    def __getitem__(self, index) -> Optional[Feature]:
        if index in self.ordinal_features:
            return self.ordinal_features[index]
        if index in self.categorical_features:
            return self.categorical_features[index]
        return None
    
    def _sample_feature(self, feature: Feature) -> Union[int, float]:
        if isinstance(feature, DiscreteFeature):
            return self._sample_discrete_feature(feature)
        elif isinstance(feature, ContinuousFeature):
            return self._sample_continuous_feature(feature)
        assert False and "Features should be discrete or continuous"
        
    def _sample_continuous_feature(self, feature: ContinuousFeature) -> float:
        sampling = getattr(self, "sampling", None)
        if sampling is None:
            sampling = FloatRandomSampling()
        problem = Problem(
            n_var=1,
            xl=feature.lb,
            xu=feature.ub,
        )
        sample = sampling.do(problem, 1).get("X")
        return sample.item()

    def _sample_discrete_feature(self, feature: DiscreteFeature) -> Union[int, float]:
        feature_type = getattr(feature, "feature_type", None)
        if feature.distribution is None or feature.distribution == "uniform":
            distribution = np.ones_like(feature.values, dtype=float) / feature.num_values
        else:
            distribution = np.array(feature.distribution) / np.sum(feature.distribution)

        category = np.random.choice(list(range(feature.num_values)), p=distribution)
        if feature_type == FeatureType.CATEGORICAL:
            return category
        
        bin_size = 1 / feature.num_values
        val = (category + 0.5) * bin_size
        val += (np.random.random() - 0.5) * bin_size
        return val

    def _sample_features(self, features: Iterable[Feature]) -> List:
        result = []
        for feature in features:
            result.append(self._sample_feature(feature))
        return result

    def sample_feature_scores(self) -> CombinedFeaturesInstance:
        return CombinedFeaturesInstance(
            ordinal=self._sample_features(self.ordinal_features.values()),
            categorical=self._sample_features(self.categorical_features.values()),
            continuouos=self._sample_features(self.continuous_features.values())
        )

    def map_categorical_indices_to_labels(self, categorical_feature_indices: List[int]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for idx, (name, feature) in zip(categorical_feature_indices, self.categorical_features.items()):
            result[name] = feature.values[idx]
        return result

    def map_numerical_scores_to_labels(self, numerical_feature_scores: List[float]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for score, (name, feature) in zip(numerical_feature_scores, self.ordinal_features.items()):
            bin_idx = int(score * feature.num_values)
            if bin_idx == feature.num_values:
                bin_idx -= 1
            result[name] = feature.values[bin_idx]
        return result
    
    def get_continuous_values_dict(self, continuouos_feature_values: List[float]) -> Dict[str, Any]:
        result = {}
        for feature_name, value in zip(self.continuous_features.keys(), continuouos_feature_values):
            result[feature_name] = value
        return result
    
    def get_feature_values_dict(
            self,
            ordinal_feature_scores: Optional[List[float]] = None,
            categorical_feature_indices: Optional[List[int]] = None,
            continuous_feature_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        result = {}
        if categorical_feature_indices is not None:
            result.update(self.map_categorical_indices_to_labels(categorical_feature_indices))
        if ordinal_feature_scores is not None:
            result.update(self.map_numerical_scores_to_labels(ordinal_feature_scores))
        if continuous_feature_values is not None:
            result.update(self.get_continuous_values_dict(continuous_feature_values))
        return result
    
    def get_var_from_feature_value(
            self,
            feature: Feature,
            value: Any,
            feature_type: FeatureType,
    ) -> Optional[Union[int, float]]:
        if value not in feature.values:
            return None
        idx = feature.values.index(value)
        if feature_type == FeatureType.CATEGORICAL:
            return idx
        if feature_type == FeatureType.ORDINAL:
            bin_size = 1 / feature.num_values
            score = (idx + 0.5) * bin_size
            score += (np.random.random() - 0.5) * bin_size
            return score
        
    def get_feature_by_name(self, name: str) -> Optional[Feature]:
        """Return feature object by name, or None if not found."""
        return (
            self.ordinal_features.get(name)
            or self.categorical_features.get(name)
            or self.continuous_features.get(name)
        )