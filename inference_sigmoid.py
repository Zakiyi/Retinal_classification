import os
import torch
import pprint
import json
import datetime
import argparse
from glob import glob
import numpy as np
import torch.nn.functional as F
from models import create_model
from collections import OrderedDict
from data_proc.augment import Augmentations
from data_proc.csv_dataset_test import CSV_Test_Dataset
from data_proc.csv_ext_dataset import CSV_Ext_Dataset
from misc.utils import calculate_metrics
from sklearn.metrics.ranking import _binary_clf_curve


class_name = {'epiretinal membrane': 0, 'macular edema': 1, 'macular hole': 2, 'myopic maculopathy': 3,
              'optic disc atrophy': 4, 'optic disc edema': 5, 'rare': 6, 'retinal detachment': 7,
              'retinal vein occlusion': 8, 'subretinal hemorrhage': 9}

# class_name = {'epiretinal membrane': 0, 'healthy': 1, 'macular edema': 2, 'macular hole': 3, 'myopic maculopathy': 4,
#               'optic disc atrophy': 5, 'optic disc edema': 6, 'rare': 7, 'retinal detachment': 8,
#               'retinal vein occlusion': 9, 'subretinal hemorrhage': 10}


def specificity_sensitivity_curve(y_true, probas_pred):
    """
    Compute specificity-sensitivity pairs for different probability thresholds.
    For reference, see 'precision_recall_curve'
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred)
    sensitivity = tps / tps[-1]
    specificity = (fps[-1] - fps) / fps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[specificity[sl], 1], np.r_[sensitivity[sl], 0], thresholds[sl]


def inference(args):
    checkpoint = torch.load(args.checkpoint)
    for k in checkpoint['args'].__dict__.keys():
        if 'data_dir' not in k:
            args.__setattr__(k, checkpoint['args'].__dict__[k])
    pprint.pprint(args.__dict__)

    aug_parameters = OrderedDict({'rotate': False,
                                  'hflip': None,
                                  'vflip': None,
                                  'color_trans': None,
                                  'coarse_dropout': None,
                                  'gama': False,
                                  'blur': False,
                                  'normalization': {'mean': (0.485, 0.456, 0.406),
                                                    'std': (0.229, 0.224, 0.225)},
                                  'size': 320,
                                  'scale': (1.0, 1.0),
                                  'ratio': (1.0, 1.0)
                                  }
                                 )

    augmentor = Augmentations(augs=aug_parameters, tta='ten')
    dataset = CSV_Ext_Dataset(data_root=args.data_dir, is_train=False, test_mode=True, fold=args.fold,
                              transform=augmentor.ta_transform)
    # dataset = CSV_Test_Dataset(data_root=args.data_dir, transform=augmentor.ta_transform, class_name=args.class_name)
    model_state_dict = checkpoint['model_state_dict']
    model = create_model(args.backbone, args.num_class, freeze_bn=args.freeze_bn, rep_bn=args.repbn).to(args.device)
    model.load_state_dict(model_state_dict)
    model.eval()
    actuals = []
    pred_scores = []
    pred_labels = []
    thresholds = 0.5*torch.ones(args.num_class).to(args.device)
    results = {}
    with torch.no_grad():
        for data, label, idx in iter(dataset):
            data = data.to(args.device)
            label = F.one_hot(label, args.num_class)
            pred_score = model(data).sigmoid().mean(dim=0)
            pred_label = torch.as_tensor(pred_score > thresholds, dtype=torch.int)
            actuals.append(label)
            pred_scores.append(pred_score)
            pred_labels.append(pred_label)

            results[idx] = {'img_name': dataset.images[idx],
                            'pred_score': pred_score.cpu().numpy().tolist(),
                            'pred_label': pred_label.cpu().numpy().tolist(),
                            'true_label': label.cpu().numpy().tolist()}

    pred_scores = torch.stack(pred_scores, dim=0).cpu().numpy()
    pred_labels = torch.stack(pred_labels, dim=0).cpu().numpy()
    true_labels = torch.stack(actuals, dim=0)

    result_metrics = {}
    for i, name in enumerate(args.class_name):
        print(i, name)
        if len(glob(os.path.join(args.data_dir, name, '*'))) > 0:
            acc, auc, f1_score, precision, recall, specificity = calculate_metrics(pred_scores[:, i], pred_labels[:, i],
                                                                                   true_labels[:, i])

            result_metrics[name] = {'auc': auc, 'acc': acc, 'f1-score': f1_score, 'precision': precision,
                                    'recall': recall, 'specificity': specificity}
            print(result_metrics[name])

            result_metrics['sp_se_thres_{}'.format(name)] = specificity_sensitivity_curve(true_labels[:, i], pred_scores[:, i])

    with open(os.path.join(args.log_dir, 'inference_{}.json'.
            format(datetime.datetime.now().strftime('%Y%m%d'))), 'w') as f:
        f.write(json.dumps(results, indent=2))

    torch.save(result_metrics, os.path.join(args.log_dir, 'result_metric.pt'))
    return result_metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data with CNN-Transformer model')
    # data
    parser.add_argument('--data-dir', default='/media/zyi/litao/Retina_recognition/ext_data/common10', help='data directory')
    parser.add_argument('--csv-file', default=None, help='path to csv file')
    parser.add_argument('--checkpoint', default='checkpoints', help='where to store checkpoints')
    parser.add_argument('--log-dir', default='runs/exp', help='where to store results')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')
    # model setting
    parser.add_argument('--backbone', default='resnet34', type=str, help='name of backbone network')
    parser.add_argument('--image-size', type=int, default=320, help='image size to the model')
    inference(parser.parse_args())
