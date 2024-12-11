import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings

from tqdm import tqdm

import numpy as  np

import torch
from torch.utils.data import Dataset, DataLoader
# from torchrec import JaggedTenso

#python3 /home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us/hierarchy_feat.py

ds_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/extract_plot_cv/0"
# ds_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/lesser_indices_correct"
ds_test_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/extract_plot_test"

ds_train = tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP")
ds_val = tf.data.Dataset.load(f"{ds_path}/validation", compression="GZIP")
ds_test = tf.data.Dataset.load(f"{ds_test_path}", compression="GZIP")


##OLD STATE

import std_models_jitter2, std_models_jitter3
import std_models_jitter_binary


from std_train_jitter import get_mean_std_dict
# Check the scenario
model = std_models_jitter_binary.Baseline_cbr_mb()

# Compute normalization values
model.set_mean_std_scores(
    get_mean_std_dict(
        tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP"),
        model.mean_std_scores_fields,
    )
)

model.load_weights("/home/ec2-user/ckpt/moblus_mb/17-0.0000")


model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics= ['accuracy'],
    # run_eagerly=False,
)





model_greater = std_models_jitter2.Baseline_cbr_mb()

# Compute normalization values
model_greater.set_mean_std_scores(
    get_mean_std_dict(
        tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP"),
        model_greater.mean_std_scores_fields,
    )
)

model_greater.load_weights("/home/ec2-user/ckpt/mb_transformer2/113-23.9258")


model_greater.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss = tf.keras.losses.MeanAbsolutePercentageError(),
    metrics= ['MeanAbsolutePercentageError'],
    # run_eagerly=False,
)



model_lesser = std_models_jitter3.Baseline_cbr_mb()

# Compute normalization values
model_lesser.set_mean_std_scores(
    get_mean_std_dict(
        tf.data.Dataset.load(f"{ds_path}/training", compression="GZIP"),
        model_lesser.mean_std_scores_fields,
    )
)

model_lesser.load_weights("/home/ec2-user/ckpt/mb_transformer2/117-11.7174")


model_lesser.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss = tf.keras.losses.MeanAbsolutePercentageError(),
    metrics= ['MeanAbsolutePercentageError'],
    # run_eagerly=False,
)

# @tf.function
def _default_individual_prediction(model: tf.keras.Model, sample: any) -> np.ndarray:
    # Obtain the prediction as numpy array, and flatten
    pred = model(sample).numpy().reshape((-1,))

    # Transform the prediction from ms to s, and return
    return  pred



train_list = []
x_values = []
labels = []

i = 0
all_preds  = []
# X_graphs = []
for item in ds_test:
    # Assuming the elements in the dataset are tensors, convert them into a format compatible with PyTorch

    predicted_masks= _default_individual_prediction(model, item[0])
    predicted_greater_delay= _default_individual_prediction(model_greater, item[0])
    # print("predicted_greater_delay", predicted_greater_delay)
    predicted_lesser_delay= _default_individual_prediction(model_lesser, item[0])
    # print("predicted_lesser_delay", predicted_lesser_delay)

    mask = np.array(predicted_masks.round(), dtype=bool)
    # print("mask", mask)
    greater_indices = np.arange(len(predicted_masks))[mask]
    lesser_indices = np.arange(len(predicted_masks))[~mask]

    delays = np.zeros(len(predicted_masks),)
    delays[greater_indices] = predicted_greater_delay[greater_indices]
    delays[lesser_indices] = predicted_lesser_delay[lesser_indices]
    all_preds.extend(delays)
    # print("delays", delays)
pd.DataFrame({"hierarchical_pred":all_preds}).to_csv("hierarchical_pred.csv")
   
    # greater_indices = np.where(predicted_delay >= 0.5)
    # lesser_indices = np.where(predicted_delay < 0.5)
    # item[0]["greater_indices"] = greater_indices
    # item[0]["lesser_indices"] = lesser_indices

    # # print(i)
    # # train_list.append((item[0], item[1], item[2].numpy(), item[3].numpy()))
    # i+=1
    # # if i >10:
    # #     break






# train_list = []
# x_values = []
# labels = []
# # X_graphs = []
# for item in ds_val:
#     # Assuming the elements in the dataset are tensors, convert them into a format compatible with PyTorch
#     greater_indices = item[0]["lesser_indices"]
#     delay = item[1]
#     # delay = tf.gather(delay, tf.transpose(greater_indices))
#     print("delay", delay)
#     # train_list.append((item[0], delay, item[2].numpy(), item[3].numpy()))

# def _generator(
#      shuffle: bool, verify_delays: bool
# ):
#     for ret in iter(train_list):
#         print(ret[1].shape)
#         # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
#         # if verify_delays and not all(x > 0 for x in ret[1]):
#         #     continue
#         yield ret

# def input_fn( shuffle: bool = False, verify_delays:bool = True) -> tf.data.Dataset:
#     """Returns a tf.data.Dataset object with the dataset stored in the given path

#     Parameters
#     ----------
#     data_dir : str
#         Path to the dataset
#     shuffle : bool, optional
#         True to shuffle the samples, False otherwise, by default False
#     verify_delays: bool, optional
#         True so that samples with unvalid delay values are discarded, by default True

#     Returns
#     -------
#     tf.data.Dataset
#         The processed dataset
#     """
#     signature = (
#         {
#             "sample_file_name": tf.TensorSpec(shape=(None,), dtype=tf.string),
#             "sample_file_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
#             "max_link_load": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "global_losses": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "global_delay": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
#             "flow_traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_bitrate_per_burst": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_tos": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_p10PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_p20PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_p50PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_p80PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_p90PktSize": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "ibg": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_variance": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_pkts_per_burst": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),            
#             "flow_packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_packet_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "flow_type": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
#             "flow_length": tf.TensorSpec(shape=(None,), dtype=tf.int32),
#             "link_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "link_capacity_and_node_type": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),

#             "devices": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32),
#             "link_to_path": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
#             "path_to_link": tf.RaggedTensorSpec(
#                 shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
#             ),
#             "flow_ipg_mean": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "greater_indices": tf.TensorSpec(shape=(1, None), dtype=tf.int32),
#             "lesser_indices": tf.TensorSpec(shape=(1, None), dtype=tf.int32),


#             # "flow_first_minute_packets": tf.RaggedTensorSpec(shape=(None,None), dtype=tf.float32),
#             "flow_ipg_var": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#             "rate": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),

#         },
#         tf.TensorSpec(shape=None, dtype=tf.float32),
#         tf.TensorSpec(shape=None, dtype=tf.float32),
#         tf.TensorSpec(shape=None, dtype=tf.float32),
#     )

#     ds = tf.data.Dataset.from_generator(
#         _generator,
#         args=[shuffle, verify_delays],
#         output_signature=signature,
#     )

#     ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

#     return ds


# tf.data.Dataset.save(
#     input_fn(
#         shuffle=True,
#         verify_delays=True,
#     ),
#    "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/lesser_indices_correct/validation",
#     compression="GZIP",
# )
