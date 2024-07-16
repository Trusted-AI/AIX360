import numpy as np
import pandas as pd
import sklearn
import shap, lime
from lime.lime_tabular import LimeTabularExplainer


def load_fico_dataset(path='./datasets/heloc-clean-full.csv'):
    data = pd.read_csv(path)
    target = 'RiskPerformance'
    return data.drop(columns=[target]), data[target]


def get_LIME_classifier(lime_expl, label_x0, x0):
    """ returns a functional (callable) form of lime explainer computed at x0.
    (code adapted from https://github.com/amparore/leaf)

    Args:
        lime_expl (_type_): returned object from lime explainer.explain_instance
        label_x0 (_type_): model predicted label of x0
        x0 (numpy array): the instance (standardized)

    Returns:
        Callable, sklearn ridge regression: linear classifier for of lime explanation
    """
    features_weights = [x[1] for x in lime_expl.local_exp[label_x0]]
    features_indices = [x[0] for x in lime_expl.local_exp[label_x0]]
    intercept = lime_expl.intercept[label_x0]
    coef = np.zeros(len(x0))
    coef[features_indices] = features_weights
    if hasattr(lime_expl, 'perfect_local_concordance') and lime_expl.perfect_local_concordance:
        # print('have perfect_local_concordance classifier!')
        g = lime.lime_base.TranslatedRidge(alpha=1.0)
        g.x0 = np.zeros(len(x0))
        g.x0 = lime_expl.x0
        # g.x0[features_indices] = lime_expl.x0[features_indices]
        g.f_x0 = lime_expl.predict_proba[label_x0]
        g.coef_ = g.ridge.coef_ = coef
        g.intercept_ = g.ridge.intercept_ = intercept
        # print('g.x0', g.x0)
        # print('g.f_x0', g.f_x0)
        # print('g.coef_', g.coef_)
        # print('g.intercept_', g.intercept_)
    else:
        g = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)#, normalize=False)
        g.coef_ = coef
        g.intercept_ = intercept
    return g

# Build the linear classifier of a SHAP explainer
def get_SHAP_classifier(label_x0, phi, phi0, x0, EX):
    """ returns a functional (callable) form of shap explainer computed at x0.
    (code adapted from https://github.com/amparore/leaf)

    Args:
        label_x0 (_type_): model predicted label of x0
        phi (_type_): _description_
        phi0 (_type_): _description_
        x0 (_type_): the instance (standardized)
        EX (_type_): _description_

    Returns:
        Callable, sklearn ridge regression: linear classifier form of shap explanation
    """
    coef = np.divide(phi[label_x0], (x0 - EX).values, where=(x0 - EX).values!=0)
    g = sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)#, normalize=False)
    g.coef_ = coef
    g.intercept_ = phi0[label_x0]
    return g


def compute_lime_explainer(x_train, model, x0,
                           # lime explanation kwargs
                           explanation_samples=1000,
                           num_features=5,
                           top_labels=1
                           ):
    """wrapper to compute lime explanation on x0 for model, note that we don't need the ground truth here.

    Args:
        x_train (_type_): standardized training data
        model (_type_): blackbox model (sklearn interface)
        x0 (_type_): instance to be explained
        explanation_samples (int, optional): _description_. Defaults to 1000.
        num_features (int, optional): _description_. Defaults to 5.
        top_labels (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: functional form of lime explainer and lime explanation
    """
    
    num_features = num_features
    top_labels = top_labels
    explanation_samples = explanation_samples

    # Get the output of the black-box classifier on x0
    output = model.predict_proba([x0])[0]
    label_x0 = np.argmax(output)  # predicted label
    prob_x0 = output[label_x0]    # predicted probability for the predicted label

    LIMEEXPL = LimeTabularExplainer(x_train.astype('float'),  # original x_train
                                    feature_names=x_train.columns.tolist(),
                                    # class_names=['0', '1'],
                                    discretize_continuous=False,
                                    sample_around_instance=True,
                                    random_state=1234)

    lime_expl = LIMEEXPL.explain_instance(np.array(x0),
                                          model.predict_proba,
                                          num_features=num_features,
                                          top_labels=top_labels,  # explain the top-1 label (by confidence) from prediction
                                          num_samples=explanation_samples)

    # get lime explanation in functional form
    func = get_LIME_classifier(lime_expl, label_x0, x0)
    return func, lime_expl


def compute_shap_explainer(x_train, model, x0, explanation_samples=1000):
    """wrapper to compute shap explanation on x0 for model, note that we don't need the ground truth here.

    Args:
        x_train (_type_): standardized training data
        model (_type_): blackbox model (sklearn interface)
        x0 (_type_): instance to be explained
        explanation_samples (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: functional form of kernel shap explainer and shap explanation
    """
    # background values for shape
    EX = x_train.mean(0)
    npEX = np.array(EX)  # all zeroes
    StdX = np.array(x_train.std(0))  # all ones
    explanation_samples = explanation_samples

    # Get the output of the black-box classifier on x0
    output = model.predict_proba([x0])[0]
    label_x0 = np.argmax(output)  # predicted label, used later in obtaining the func shap
    prob_x0 = output[label_x0]  # predicted probability for the predicted label

    # shap
    SHAPEXPL = shap.KernelExplainer(model.predict_proba,
                                    EX,
                                    nsamples=explanation_samples)

    shap_phi = SHAPEXPL.shap_values(x0, l1_reg="num_features(10)")
    shap_phi0 = SHAPEXPL.expected_value

    func = get_SHAP_classifier(label_x0, shap_phi, shap_phi0, x0, EX)
    return func, (shap_phi, shap_phi0)


# RISE explanation utilities
from tqdm.auto import tqdm
from skimage.transform import resize

def generate_masks(N, s, p1, input_size=(32, 32)):
    """
    N: no. of masks
    p1: some probability?
    s: ?
    """
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float')

    masks = np.empty((N, *input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    masks = masks.reshape(-1, *input_size, 1)
    return masks


def explain(batch_predict, inp, masks, N=2000, batch_size=100, input_size=(32, 32), p1=0.5):
    preds = []
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, N, batch_size), desc='Explaining'):
        preds.append(batch_predict(masked[i:min(i+batch_size, N)]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *input_size)
    sal = sal / N / p1
    return sal
