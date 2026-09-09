import sys
import torch
from sklearn.manifold import TSNE
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def scatter(features, targets, workdir, fig_name, subtitle=None, n_classes=16):
    targets = targets.reshape(targets.shape[0], )
    palette = np.array(sns.color_palette("hls", n_classes))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    save_path = f"{workdir}/{fig_name}.png"
    plt.savefig(save_path, dpi=600)
    # plt.show()


def obtain_embedding_feature_map(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            if torch.cuda.is_available():
                data = data.float().to(device)
                target = target.long().to(device)

            _, feature = model(data)

            feature_map[len(feature_map):len(feature) - 1] = feature.tolist()
            target_output[len(target_output):len(target) - 1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output


def get_accuracy_score(y_true, y_pred, logger):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    acc = sum(w[ind[0][idx], ind[1][idx]] for idx in range(ind[0].size)) * 1.0 / y_pred.size
    logger.info(f"ACC↑ = {acc:.8f}")
    return acc


def get_silhouette_score(x_test_feature, y_test, logger):
    sc = metrics.silhouette_score(x_test_feature, y_test)
    logger.info(f"SC ↑ = {sc:.8f}")


def get_calinski_score(x_test_feature, y_test, logger):
    ch = metrics.calinski_harabasz_score(x_test_feature, y_test)
    logger.info(f"CHI↑ = {ch:.8f}")


def get_davies_score(x_test_feature, y_test, logger):
    dbi = metrics.davies_bouldin_score(x_test_feature, y_test)
    logger.info(f"DBI↓ = {dbi:.8f}\n")


def get_classification_report(x_test_feature, y_test):
    classification_report = metrics.classification_report(x_test_feature, y_test)
    logger.info(f"classification_report = {classification_report:.8f}")


def visualize(model, test_dataloader, num_class, device, logger, workdir, fig_name):
    model = model.to(device)
    X_test_embedding_feature_map, real_target = obtain_embedding_feature_map(model, test_dataloader, device)

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    scatter(eval_tsne_embeds, real_target.astype('int64'), workdir, fig_name, "ETTA", num_class)

    km = KMeans(n_clusters=num_class, n_init=30)
    km.fit(eval_tsne_embeds)
    cluster_target = km.predict(eval_tsne_embeds)

    get_silhouette_score(X_test_embedding_feature_map, real_target, logger)
    get_calinski_score(X_test_embedding_feature_map, real_target, logger)
    logger.info(f"NMI↑ = {metrics.normalized_mutual_info_score(real_target.squeeze(), cluster_target)}")
    logger.info(f"AMI↑ = {metrics.adjusted_rand_score(real_target.squeeze(), cluster_target)}")
    get_accuracy_score(real_target, cluster_target, logger)
    get_davies_score(X_test_embedding_feature_map, real_target, logger)
