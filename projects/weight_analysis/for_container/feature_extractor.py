import os
import numpy as np

DATA_PATH = '/scratch/data/TrojAI/cyber-pdf-dec2022-train/models/'


def get_features_and_labels(model_repr_dict: dict, model_ground_truth_dict: dict):
    X, y = [], []

    for model_arch, model_reprs in model_repr_dict.items():
        model_ground_truth = model_ground_truth_dict[model_arch]
        y += model_ground_truth

        for model_repr in model_reprs:
            X.append(get_model_features(model_repr, infer=False))

    return np.asarray(X), np.asarray(y)


def get_model_features(model_repr : dict, infer=True):
    features = []
    features += _get_weight_features(model_repr)
    features += _get_eigen_features(model_repr)
    if infer:
        return np.asarray([features])
    else:
        return features


def _get_weight_features(model_repr : dict, layers=['fc1.weight', 'fc1.bias'], axis=0) -> list:
    params = []
    for layer in layers:
        param = model_repr[layer]
        if len(param.shape) > 1:
            params += np.amax(param, axis=axis).tolist()
            params += np.mean(param, axis=axis).tolist()
            sub = np.mean(param, axis=axis) - np.median(param, axis=axis)
            params += sub.tolist()
            params += np.median(param, axis=axis).tolist()
            params += np.sum(param, axis=axis).tolist()
            params.append(np.linalg.norm(param, ord='fro')**2/np.linalg.norm(param, ord=2)**2)
        else:
            params.append(param.max().tolist())
            params.append(param.mean().tolist())
            sub = param.mean() - np.median(param)
            params.append(sub.tolist())
            params.append(np.median(param).tolist())
            params.append(param.sum().tolist())
            params.append((np.linalg.norm(param.reshape(param.shape[0], -1), ord='fro')**2/np.linalg.norm(param.reshape(param.shape[0], -1), ord=2)**2).tolist())
    return params


def _get_eigen_features(model_repr: dict) -> list:
    fc1_weight = model_repr['fc1.weight'].T.reshape(135, 10, 10)
    _, s, _ = np.linalg.svd(fc1_weight)
    return s.flatten().tolist()


if __name__ =='__main__':
    from utils.models import load_models_dirpath

    models_dirpath = sorted([os.path.join(DATA_PATH, model_dirpath) for model_dirpath in os.listdir(DATA_PATH)])
    model_repr_dict, model_ground_truth_dict = load_models_dirpath(models_dirpath)
    X, y = get_features_and_labels(model_repr_dict, model_ground_truth_dict)

    OUTPUT_DIR = '/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/extracted_source'
    np.save(os.path.join(OUTPUT_DIR, 'fe_X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'fe_y.npy'), y)