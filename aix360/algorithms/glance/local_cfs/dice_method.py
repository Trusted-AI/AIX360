from ..base import LocalCounterfactualMethod
import dice_ml
import pandas as pd


class DiceMethod(LocalCounterfactualMethod):
    """
    Implementation of the Dice method for generating counterfactual instances.
    """

    def __init__(self):
        super().__init__()
        self.cf_generator = None

    def fit(
        self,
        model,
        data,
        outcome_name,
        continuous_features,
        feat_to_vary,
        random_seed=13,
    ):
        dice_dataset = dice_ml.Data(
            dataframe=data,
            continuous_features=continuous_features,
            outcome_name=outcome_name,
        )
        self.random_seed = random_seed
        self.feat_to_vary = feat_to_vary
        dice_model = dice_ml.Model(model=model, backend="sklearn", func=None)
        self.cf_generator = dice_ml.Dice(dice_dataset, dice_model, method="random")

    def explain_instances(
        self, instances: pd.DataFrame, num_counterfactuals: int
    ) -> pd.DataFrame:
        if self.cf_generator is None:
            raise ValueError("Fit the Local Counterfactual method first.")

        counterfactuals = self.cf_generator.generate_counterfactuals(
            instances,
            total_CFs=num_counterfactuals,
            desired_class=1,
            random_seed=self.random_seed,
            features_to_vary=self.feat_to_vary,
            posthoc_sparsity_param=None,
        )

        return pd.concat(
            [
                counterfactuals.cf_examples_list[i].final_cfs_df.iloc[:, :-1]
                for i in range(len(instances))
            ],
            ignore_index=False,
        )
