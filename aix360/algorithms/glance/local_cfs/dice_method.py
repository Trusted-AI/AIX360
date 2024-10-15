from ..base import LocalCounterfactualMethod
import dice_ml
import pandas as pd


class DiceMethod(LocalCounterfactualMethod):
    """
    Implementation of the Dice method for generating counterfactual instances.(https://interpret.ml/DiCE/)

    The Dice method uses a specified machine learning model and data to generate counterfactual examples,
    providing insights into how changes in feature values can influence model predictions.

    Methods:
    --------
    __init__():
        Initializes the DiceMethod instance.

    fit(model, data, outcome_name, continuous_features, feat_to_vary, random_seed=13):
        Fits the DiceMethod to the provided dataset, preparing the counterfactual generator.

    explain_instances(instances, num_counterfactuals):
        Generates counterfactual instances for the specified input instances.
    """

    def __init__(self):
        """
        Initializes a new instance of the DiceMethod class.
        
        Attributes:
        ----------
        cf_generator : None or dice_ml.Dice
            Counterfactual generator instance, initially set to None.
        """
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
        """
        Fits the DiceMethod to the provided dataset by creating a counterfactual generator.

        Parameters:
        ----------
        model : object
            A machine learning model used for predictions.
        data : pd.DataFrame
            The dataset containing features and the outcome variable.
        outcome_name : str
            The name of the outcome variable in the dataset.
        continuous_features : List[str]
            A list of names for continuous (numerical) features.
        feat_to_vary : List[str]
            A list of feature names that can be varied to generate counterfactuals.
        random_seed : int, optional
            Seed for random number generation to ensure reproducibility, by default 13.
        """
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
        """
        Generates counterfactual instances for the specified input instances.

        Parameters:
        ----------
        instances : pd.DataFrame
            DataFrame containing the instances for which counterfactuals are generated.
        num_counterfactuals : int
            The number of counterfactuals to generate for each instance.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the generated counterfactuals.

        Raises:
        -------
        ValueError
            If the counterfactual generator has not been initialized (fit method not called).
        """
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
