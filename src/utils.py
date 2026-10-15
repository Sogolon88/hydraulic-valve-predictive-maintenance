"""Compatibilité : les notebooks importent depuis src.utils.
Le code est rangé dans src.config, src.data, src.features, src.plots et src.pipeline ;
pour du nouveau code, importer directement depuis ces modules."""
from src.config import N_TRAIN, PIPELINE_SET, PROJECT_ROOT
from src.data import (class_distribution, list_models, load_data, load_model, load_processed,
                      make_binary_target, missing_summary, save_data, save_models, split_data,
                      split_first_n)
from src.features import (FEATURE_SETS, build_feature_set, build_features, build_frequence,
                          build_histogramme, count_outliers_iqr, drop_constant_features,
                          drop_correlated_features, feature_reduction_stats, histogram_edges,
                          select_features)
from src.pipeline import cycle_features as _cycle_features
from src.pipeline import predict_cycle as pipeline
from src.plots import (plot_boxplots, plot_class_distribution, plot_correlation_matrix,
                       plot_histograms, save_figure)
