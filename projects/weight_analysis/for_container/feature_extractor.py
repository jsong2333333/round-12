import os
import numpy as np

DATA_PATH = '/scratch/data/TrojAI/cyber-pdf-dec2022-train/models/'
NET_LEVEL = 7


def get_features_and_labels(model_repr_dict: dict, model_ground_truth_dict: dict):
    X, y = {'small_net':[], 'large_net':[]}, {'small_net':[], 'large_net':[]}

    for model_arch, model_reprs in model_repr_dict.items():
        model_ground_truth = model_ground_truth_dict[model_arch]
        net = 'small_net'
        if int(model_arch[3]) > NET_LEVEL:
            net = 'large_net'

        y[net] += model_ground_truth

        for model_repr in model_reprs:
            X[net].append(get_model_features(model_repr, model_arch, infer=False))
    
    return np.asarray(X['small_net']), np.asarray(y['small_net']), np.asarray(X['large_net']), np.asarray(y['large_net'])


def get_model_features(model_repr: dict, model_class: str, infer=True):
    features = []
    ok = _get_ordered_key(model_repr)

    if int(model_class[3]) > NET_LEVEL:  #larger nets (excluding Net2)
        norm_mul_weight = _get_multiplied_weight_features(model_repr, ok, normalized=True)
        features += norm_mul_weight.flatten().tolist()
        features += _get_fft_from_weight_features(norm_mul_weight)
        mul_weight = _get_multiplied_weight_features(model_repr, ok[1:])
        features += _get_eigen_from_weight_features(mul_weight)
        features += _get_stats_from_weight_features(mul_weight)
    else:
        for k in ['fc1.weight', 'fc1.bias']:
            features += _get_stats_from_weight_features(model_repr[k])
        features += _get_multiplied_weight_features(model_repr, ok).flatten().tolist()
    if infer:
        return np.asarray([features])
    else:
        return features


def _get_ordered_key(model_repr: dict, weight_or_bias='weight', reversed=True) -> list:
    keys = [k for k in model_repr.keys() if weight_or_bias in k]
    if reversed:
        return keys[::-1]
    else:
        return keys


def _get_multiplied_weight_features(model_repr: dict, ordered_keys: list, normalized=False) -> np.ndarray:
    weight = None
    for ok in ordered_keys:
        if normalized:
            weight = model_repr[ok] / np.linalg.norm(model_repr[ok],2) if weight is None else (weight @ (model_repr[ok]/ np.linalg.norm(model_repr[ok],2)))  
        else:
            weight = model_repr[ok] if weight is None else weight @ model_repr[ok]
    return weight


def _get_stats_from_weight_features(weight: np.ndarray, axis= (0,)) -> list:
    params = []
    p_max = np.amax(weight, axis=axis) 
    p_mean = np.mean(weight, axis=axis)
    p_median = np.median(weight, axis=axis) 
    p_sub = p_mean - p_median
    p_sum = np.sum(weight, axis=axis)
    try:
        p_norm = np.linalg.norm(weight, ord='fro')**2/np.linalg.norm(weight, ord=2)**2
    except:
        p_norm = np.linalg.norm(weight.reshape(weight.shape[0], -1), ord='fro')**2/np.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2)**2
    
    if len(weight.shape) - len(axis) == 0:
        params = [p_max.tolist(), p_mean.tolist(), p_sub.tolist(), p_median.tolist(), p_sum.tolist(), p_norm]
    else:
        params = p_max.tolist() + p_mean.tolist() + p_sub.tolist() + p_median.tolist() + p_sum.tolist()
        params.append(p_norm)
    return params


def _get_fft_from_weight_features(weight: np.ndarray) -> list:
    ft = np.fft.ifftshift(weight)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    return np.abs(ft.flatten()).tolist()


def _get_eigen_from_weight_features(weight: np.ndarray, sind=0, eind=45) -> list:
    _, s, _ = np.linalg.svd(weight)
    return s[sind:eind].tolist()


if __name__ =='__main__':
    from utils.models import load_models_dirpath

    models_dirpath = sorted([os.path.join(DATA_PATH, model_dirpath) for model_dirpath in os.listdir(DATA_PATH)])
    model_repr_dict, model_ground_truth_dict = load_models_dirpath(models_dirpath)
    X_s, y_s, X_l, y_l = get_features_and_labels(model_repr_dict, model_ground_truth_dict)

    OUTPUT_DIR = '/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/extracted_source'
    np.save(os.path.join(OUTPUT_DIR, 'fe_X_s.npy'), X_s)
    np.save(os.path.join(OUTPUT_DIR, 'fe_y_s.npy'), y_s)
    np.save(os.path.join(OUTPUT_DIR, 'fe_X_l.npy'), X_l)
    np.save(os.path.join(OUTPUT_DIR, 'fe_y_l.npy'), y_l)