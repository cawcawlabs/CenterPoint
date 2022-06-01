# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

import json
import math
import numpy as np
from matplotlib import pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from scipy.spatial.transform import Rotation as R
from typing import Any
Axis = Any

def nuscenes_label_to_names():
    return {
        0: 'ignore',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck'
    }

def open3d_visualize_sample(nusc: NuScenes,
                            sample_token: str,
                            gt_boxes: EvalBoxes,
                            pred_boxes: EvalBoxes,
                            nsweeps: int = 1,
                            eval_range: float = 50,
                            verbose: bool = True,
                            savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: Path to save the summary writer to.
    """
    try:
        import tensorflow
        import open3d.ml.tf as ml3d
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        print("Could not import open3d.ml.tf or torch.utils.tensorboard. \
            Please run pip install -r {nuscenes-devkit_path}/setup/requirements/requirements_open3d_ml.txt to visualize 3d plots.")
        return
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    lidar2ego_rotation = cs_record['rotation']
    lidar2ego_translation = cs_record['translation']
    world_cam = np.eye(4)
    world_cam[:3, :3] = R.from_quat(lidar2ego_rotation).as_matrix()
    world_cam[:3, -1] = lidar2ego_translation
    world_cam = world_cam.T

    # Get boxes.
    if gt_boxes and len(gt_boxes):
        boxes_gt_global = gt_boxes[sample_token]
    else:
        boxes_gt_global = []
    if pred_boxes and len(pred_boxes):
        boxes_est_global = pred_boxes[sample_token]
    else:
        boxes_est_global = []

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_pred = boxes_to_sensor(boxes_est_global, pose_record, cs_record)
    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
    points = pc.points.T

    step_counter = 1
    print("Open3d plot to:", savepath)
    writer = SummaryWriter(savepath)
    writer.add_3d(
        "input_pointcloud",
        {  # float, shape (N, 3), or (B, N, 3) for a batch
            "vertex_positions": points[:, :3],
            # Extra features: float, shape (N, 1), or (B, N, 1) for a batch
            # [should not be (N,)]
            "vertex_intensities": points[:, 3].reshape(-1, 1),
            # "vertex_ring_indicies": pc[:, 4].reshape(-1, 1),
        },
        step_counter
    )
    label_to_names = nuscenes_label_to_names()
    names_to_labels = dict((v, k) for k, v in label_to_names.items())
    gt_boxes = nuscenes_boxes_to_open3d_bev_boxes(
        world_cam=world_cam,
        boxes=boxes_gt,
        names_to_labels=names_to_labels
    )
    if len(gt_boxes):
        writer.add_3d(
            "object_detection_gt",
            {"bboxes": gt_boxes},
            step_counter,
            label_to_names=label_to_names
        )
    pred_boxes = nuscenes_boxes_to_open3d_bev_boxes(
        world_cam=world_cam,
        boxes=boxes_pred,
        names_to_labels=names_to_labels
    )
    if len(pred_boxes):
        writer.add_3d(
            "object_detection_pred",
            {"bboxes": pred_boxes},
            step_counter,
            label_to_names=label_to_names
        )

def nuscenes_boxes_to_open3d_bev_boxes(world_cam, boxes, names_to_labels):
    objects = []
    for box in boxes:
        center = box.center
        size = np.array([box.wlh[0], box.wlh[2], box.wlh[1]])
        label = names_to_labels[box.name]
        yaw = (box.orientation.radians - (math.pi / 2)) % (2 * math.pi)  # rotate box 90 degrees counter-clockwise.
        yaw = 2 * math.pi - yaw  # flip across the x-axis
        objects.append(
            ml3d.datasets.utils.BEVBox3D(
                center=center,
                size=size,
                yaw=yaw,
                label_class=label,
                confidence=box.score,
                show_confidence=True,
                world_cam=world_cam
            )
        )
    return objects


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close(fig)
        plt.close('all')
    else:
        plt.show()


def class_pr_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot a precision recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Precision', xlim=1,
                        ylim=1, min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def dist_pr_curve(md_list: DetectionMetricDataList,
                  metrics: DetectionMetrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None) -> None:
    """
    Plot the PR curves for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param dist_th: Distance threshold for matching.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel='Precision',
                    xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall, ax=ax)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{}: {:.1f}%'.format(PRETTY_DETECTION_NAMES[detection_name], ap * 100),
                color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def summary_plot(md_list: DetectionMetricDataList,
                 metrics: DetectionMetrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(PRETTY_DETECTION_NAMES[detection_name]), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,  min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def detailed_results_table_tex(metrics_path: str, output_path: str) -> None:
    """
    Renders a detailed results table in tex.
    :param metrics_path: path to a serialized DetectionMetrics file.
    :param output_path: path to the output file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    tex = ''
    tex += '\\begin{table}[]\n'
    tex += '\\small\n'
    tex += '\\begin{tabular}{| c | c | c | c | c | c | c |} \\hline\n'
    tex += '\\textbf{Class}    &   \\textbf{AP}  &   \\textbf{ATE} &   \\textbf{ASE} & \\textbf{AOE}   & ' \
           '\\textbf{AVE}   & ' \
           '\\textbf{AAE}   \\\\ \\hline ' \
           '\\hline\n'
    for name in DETECTION_NAMES:
        ap = np.mean(metrics['label_aps'][name].values()) * 100
        ate = metrics['label_tp_errors'][name]['trans_err']
        ase = metrics['label_tp_errors'][name]['scale_err']
        aoe = metrics['label_tp_errors'][name]['orient_err']
        ave = metrics['label_tp_errors'][name]['vel_err']
        aae = metrics['label_tp_errors'][name]['attr_err']
        tex_name = PRETTY_DETECTION_NAMES[name]
        if name == 'traffic_cone':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase)
        elif name == 'barrier':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase, aoe)
        else:
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
                   '\\hline\n'.format(tex_name, ap, ate, ase, aoe, ave, aae)

    map_ = metrics['mean_ap']
    mate = metrics['tp_errors']['trans_err']
    mase = metrics['tp_errors']['scale_err']
    maoe = metrics['tp_errors']['orient_err']
    mave = metrics['tp_errors']['vel_err']
    maae = metrics['tp_errors']['attr_err']
    tex += '\\hline {} &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
           '\\hline\n'.format('\\textbf{Mean}', map_, mate, mase, maoe, mave, maae)

    tex += '\\end{tabular}\n'

    # All one line
    tex += '\\caption{Detailed detection performance on the val set. \n'
    tex += 'AP: average precision averaged over distance thresholds (%), \n'
    tex += 'ATE: average translation error (${}$), \n'.format(TP_METRICS_UNITS['trans_err'])
    tex += 'ASE: average scale error (${}$), \n'.format(TP_METRICS_UNITS['scale_err'])
    tex += 'AOE: average orientation error (${}$), \n'.format(TP_METRICS_UNITS['orient_err'])
    tex += 'AVE: average velocity error (${}$), \n'.format(TP_METRICS_UNITS['vel_err'])
    tex += 'AAE: average attribute error (${}$). \n'.format(TP_METRICS_UNITS['attr_err'])
    tex += 'nuScenes Detection Score (NDS) = {:.1f} \n'.format(metrics['nd_score'] * 100)
    tex += '}\n'

    tex += '\\end{table}\n'

    with open(output_path, 'w') as f:
        f.write(tex)
