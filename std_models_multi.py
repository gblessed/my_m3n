"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import tensorflow as tf
import keras.backend as K

class Baseline_cbr_mb(tf.keras.Model):
    mean_std_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_pkts_per_burst",
        "flow_bitrate_per_burst",
        "flow_packet_size",
        "flow_p90PktSize",
        "rate",
        "flow_ipg_mean",
        "ibg",
        "flow_ipg_var",
        "link_capacity",
    }

    mean_std_scores = None

    name = "ufpa_ericsson_cbr_mb"

    def __init__(self, override_mean_std_scores=None, name=None):
        super(Baseline_cbr_mb, self).__init__()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delay_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="delay_mape")
        self.jitter_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="jitter_mape")
        self.pkts_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="pkts_metric")

        self.iterations = 12
        self.path_state_dim = 16
        self.link_state_dim = 16

        if override_mean_std_scores is not None:
            self.set_mean_std_scores(override_mean_std_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        self.attention = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
            tf.keras.layers.Dense(
                self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
            ),
            ]
        )

            # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate",
            ),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate",
        )

        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=13),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    )
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    )
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )

        self.jitter_readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="JitterReadout",
        )
        self.pkt_loss_readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.gelu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
            ],
            name="pkt_lossReadout",
        )    
    def set_mean_std_scores(self, override_mean_std_scores):
        assert (
            type(override_mean_std_scores) == dict
            and all(kk in override_mean_std_scores for kk in self.mean_std_scores_fields)
            and all(len(val) == 2 for val in override_mean_std_scores.values())
        ), "overriden mean-std dict is not valid!"
        self.mean_std_scores = override_mean_std_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert self.mean_std_scores is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        global_delay = inputs["global_delay"]
        global_losses = inputs["global_losses"]
        max_link_load = inputs["max_link_load"]
        flow_pkt_per_burst = inputs["flow_pkts_per_burst"]
        flow_bitrate = inputs["flow_bitrate_per_burst"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_type = inputs["flow_type"]
        flow_ipg_mean = inputs["flow_ipg_mean"]
        flow_length = inputs["flow_length"]
        ibg = inputs["ibg"]
        flow_p90pktsize = inputs["flow_p90PktSize"]
        cbr_rate = inputs["rate"]
        flow_ipg_var = inputs["flow_ipg_var"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]

        flow_pkt_size_normal = (flow_packet_size - self.mean_std_scores["flow_packet_size"][0]) \
                    * self.mean_std_scores["flow_packet_size"][1],

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
        normal_load = tf.math.divide(load, tf.squeeze(max_link_load))
        
        # Initialize the initial hidden state for paths
        path_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.mean_std_scores["flow_traffic"][0])
                    * self.mean_std_scores["flow_traffic"][1],
                    (flow_packets - self.mean_std_scores["flow_packets"][0])
                    * self.mean_std_scores["flow_packets"][1],
                    (ibg - self.mean_std_scores["ibg"][0])
                    * self.mean_std_scores["ibg"][1],
                    (cbr_rate - self.mean_std_scores["rate"][0])
                    * self.mean_std_scores["rate"][1],
                    (flow_p90pktsize - self.mean_std_scores["flow_p90PktSize"][0])
                    * self.mean_std_scores["flow_p90PktSize"][1],
                    (flow_packet_size - self.mean_std_scores["flow_packet_size"][0])
                    * self.mean_std_scores["flow_packet_size"][1],
                    (flow_bitrate - self.mean_std_scores["flow_bitrate_per_burst"][0])
                    * self.mean_std_scores["flow_bitrate_per_burst"][1],
                    (flow_ipg_mean - self.mean_std_scores["flow_ipg_mean"][0])
                    * self.mean_std_scores["flow_ipg_mean"][1],
                    (flow_ipg_var - self.mean_std_scores["flow_ipg_var"][0])
                    * self.mean_std_scores["flow_ipg_var"][1],
                    (flow_pkt_per_burst - self.mean_std_scores["flow_pkts_per_burst"][0])
                    * self.mean_std_scores["flow_pkts_per_burst"][1],
                    tf.expand_dims(tf.cast(flow_length, dtype=tf.float32), 1),
                    flow_type
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                   (link_capacity - self.mean_std_scores["link_capacity"][0])
                    * self.mean_std_scores["link_capacity"][1],
                    load,
                    normal_load,
                ],
                axis=1,
            ),
        )
        
        is_below_threshold = tf.constant(True)
        # Iterate t times doing the message passing
        for _  in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")

            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state=path_state
            )
            
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )
            
            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToLink"
            )
            
            attention_coef = self.attention(path_gather)
            normalized_score = K.softmax(attention_coef)
            weighted_score = normalized_score * path_gather
            
            path_gather_score = tf.math.reduce_sum(weighted_score, axis=1)
            
            link_state, _ = self.link_update(path_gather_score, states=link_state)


        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        # occupancy = all_reads[:,:, 0]
        # pkts_dropped = all_reads[:,:, 1]
        # jitter_occupancy = all_reads[:,:, 1]

        pkts_dropped = self.pkt_loss_readout_path(path_state)
      
     
        jitter_occupancy = self.jitter_readout_path(path_state_sequence[:, 1:])

        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        jitter_delay_sequence = jitter_occupancy / capacity_gather

        jitter = tf.math.reduce_sum(jitter_delay_sequence, axis=1)
    
        delay = tf.math.reduce_sum(delay_sequence, axis=1)
        # print("preds=", delay, jitter, pkts_dropped)s
        return delay, jitter, pkts_dropped
    
    def train_step(self, data):
        x, delay, jitter, pktsdropped = data
        # print("packets dropped ",pktsdropped[:10, ] , tf.squeeze(x["flow_traffic"], axis=1)[:10])
        pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        # print("vals===", pktsdropped)
        with tf.GradientTape() as tape:
            delay_pred, jitter_pred, pkts_pred= self(x, training=True)  # Forward pass
            # Compute our own lossMeanSquaredLogarithmicError
            # mean_squared_logarithmic_error
            loss = .33 * tf.keras.losses.mean_absolute_percentage_error(delay, delay_pred)
            loss+= .33* tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
            loss += .33* tf.keras.losses.mean_absolute_percentage_error(pktsdropped, pkts_pred)
        # loss *=.3
        # loss = (delayloss + jitterloss + pktsloss) / 3
        # Compute gradients
        trainable_vars = self.trainable_variables
        # print("lennnn ", len(trainable_vars))
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.delay_metric.update_state(delay, delay_pred)
        self.jitter_metric.update_state(jitter, jitter_pred)
        self.pkts_metric.update_state(pktsdropped, pkts_pred)

        return {"loss": self.loss_tracker.result(), "train_delay_mape": self.delay_metric.result(), "train_jitter_mape": self.jitter_metric.result(), "train_pkts_mape": self.pkts_metric.result(),}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.delay_metric, self.jitter_metric, self.pkts_metric ]
    
    def compute_loss(self, delay, delay_pred, jitter, jitter_pred, pktsdropped, pkts_pred, sample_weight= None):

        loss = .33 * tf.keras.losses.mean_absolute_percentage_error(delay, delay_pred)
        loss+= .33* tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
        loss += .33* tf.keras.losses.mean_absolute_percentage_error(pktsdropped, pkts_pred)
        self.loss_tracker.update_state(loss)
        ## increase the complexity of the model
        ## try to use sigmoid for the packets dropped
        return loss
    
    def test_step(self, data):
        # Unpack the data
        x, delay, jitter, pktsdropped  = data
        pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        performance_metrics = [delay, jitter, pktsdropped]
        # Compute predictions
        delay_pred, jitter_pred, pkts_pred = self(x, training=False)
        predictions = [delay_pred, jitter_pred, pkts_pred ]

        # Updates the metrics tracking the loss
        self.compute_loss(delay, delay_pred, jitter, jitter_pred, pktsdropped, pkts_pred)
    
        # Update the metrics.
        for i, metric in enumerate(self.metrics[1:]):
            if metric.name != "loss":
                metric.update_state(performance_metrics[i], predictions[i])
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        # total_loss ={ "val_loss" : sum( [.33* m.result() for m in self.metrics[1:]])  }
        # total_loss.update({m.name: m.result() for m in self.metrics[1:]})
        # print("total loss is ",total_loss )
        return {m.name: m.result() for m in self.metrics}
       
# class Baseline_mb(tf.keras.Model):
#     mean_std_scores_fields = {
#         "flow_traffic",
#         "flow_packets",
#         "flow_pkts_per_burst",
#         "flow_bitrate_per_burst",
#         "flow_packet_size",
#         "flow_p90PktSize",
#         "rate",
#         "flow_ipg_mean",
#         "ibg",
#         "flow_ipg_var",
#         "link_capacity",
#     }
#     mean_std_scores = None
#     def __init__(self, n_iterations=12, path_state_dim=16, link_state_dim=16, override_min_max_scores=None, name=None) -> None:
#         super(Baseline_mb, self).__init__()

#         self.iterations = n_iterations
#         self.path_state_dim = path_state_dim
#         self.link_state_dim = link_state_dim

#         if override_min_max_scores is not None:
#             self.set_min_max_scores(override_min_max_scores)
#         if name is not None:
#             assert type(name) == str, "name must be a string"
#             self.name = name

#         self.attention = tf.keras.Sequential(
#             [tf.keras.layers.Input(shape=(None, None, path_state_dim)),
#             tf.keras.layers.Dense(
#                 self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)    
#             ),
#             ]
#         )

#         # GRU Cells used in the Message Passing step
#         self.path_update = tf.keras.layers.RNN(
#             tf.keras.layers.GRUCell(self.path_state_dim,
#             name="PathUpdate"),
#             return_sequences=True,
#             return_state=True,
#             name="PathUpdateRNN",
#         )

#         self.link_update = tf.keras.layers.GRUCell(
#             self.link_state_dim, 
#             name="LinkUpdate"
#         )

#         self.path_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Input(shape=9),
#                 tf.keras.layers.Dense(
#                     self.path_state_dim, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#                 tf.keras.layers.Dense(
#                     self.path_state_dim, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#             ],
#             name="PathEmbedding",
#         )

#         self.link_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Input(shape=2),
#                 tf.keras.layers.Dense(
#                     self.link_state_dim, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#                 tf.keras.layers.Dense(
#                     self.link_state_dim, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#             ],
#             name="LinkEmbedding",
#         )

#         self.readout_path = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Input(shape=(None, self.path_state_dim)),
#                 tf.keras.layers.Dense(
#                     self.link_state_dim // 2, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#                 tf.keras.layers.Dense(
#                     self.link_state_dim // 4, activation=tf.keras.activations.selu,
#                 kernel_initializer='lecun_uniform',
#                 ),
#                 tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
#             ],
#             name="PathReadout",
#         )

#     def set_min_max_scores(self, override_min_max_scores):
#         assert (
#             type(override_min_max_scores) == dict
#             and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
#             and all(len(val) == 2 for val in override_min_max_scores.values())
#         ), "overriden min-max dict is not valid!"
#         self.min_max_scores = override_min_max_scores

#     @tf.function
#     def call(self, inputs):
#         # Ensure that the min-max scores are set
#         assert self.min_max_scores is not None, "the model cannot be called before setting the min-max scores!"
        
    
#         # Process raw inputs
#         flow_traffic = inputs["flow_traffic"]
#         flow_pkt_per_burst = inputs["flow_pkts_per_burst"]
#         flow_bitrate = inputs["flow_bitrate_per_burst"]
#         global_losses = inputs["global_losses"]
#         global_delay = inputs["global_delay"]
#         max_link_load = inputs["max_link_load"]
#         flow_length = inputs["flow_length"]
#         flow_packets = inputs["flow_packets"]
#         # flow_ipg_mean = inputs["flow_ipg_mean"]
#         # flow_ipg_var = inputs["flow_ipg_var"]
#         ibg = inputs["ibg"]
#         flow_type = inputs["flow_type"]
#         flow_packet_size = inputs["flow_packet_size"]
#         link_capacity = inputs["link_capacity"]
#         link_to_path = inputs["link_to_path"]
#         path_to_link = inputs["path_to_link"]

#         flow_pkt_size_normal = (flow_packet_size - self.min_max_scores["flow_packet_size"][0]) \
#                     * self.min_max_scores["flow_packet_size"][1],

#         path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
#         load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
#         normal_load = tf.math.divide(load, tf.squeeze(max_link_load))

#         capacity_gather = tf.gather(link_capacity, link_to_path)
#         trans_delay = flow_pkt_size_normal / tf.math.reduce_sum(capacity_gather)
#         trans_delay = tf.math.reduce_sum(trans_delay, axis=0)   


#         # Initialize the initial hidden state for paths
#         path_state = self.path_embedding(
#             tf.concat(
#                 [
#                     (flow_traffic - self.min_max_scores["flow_traffic"][0])
#                     * self.min_max_scores["flow_traffic"][1],
#                     (flow_packets - self.min_max_scores["flow_packets"][0])
#                     * self.min_max_scores["flow_packets"][1],
#                     (ibg - self.min_max_scores["ibg"][0])
#                     * self.min_max_scores["ibg"][1],
#                     (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
#                     * self.min_max_scores["flow_packet_size"][1],
#                     (flow_bitrate - self.min_max_scores["flow_bitrate_per_burst"][0])
#                     * self.min_max_scores["flow_bitrate_per_burst"][1],
#                     # (flow_ipg_mean - self.min_max_scores["flow_ipg_mean"][0])
#                     # * self.min_max_scores["flow_ipg_mean"][1],
#                     # (flow_ipg_var - self.min_max_scores["flow_ipg_var"][0])
#                     # * self.min_max_scores["flow_ipg_var"][1],
#                     (flow_pkt_per_burst - self.min_max_scores["flow_pkts_per_burst"][0])
#                     * self.min_max_scores["flow_pkts_per_burst"][1],
#                     tf.expand_dims(tf.cast(flow_length, dtype=tf.float32), 1),
#                     flow_type,
#                 ],
#                 axis=1,
#             )
#         )

#         # Initialize the initial hidden state for links
#         link_state = self.link_embedding(
#             tf.concat(
#                 [
#                     (link_capacity - self.min_max_scores["link_capacity"][0])
#                     * self.min_max_scores["link_capacity"][1],
#                     load,
#                     normal_load
#                 ],
#                 axis=1,
#             ),
#         )

#         # Iterate t times doing the message passing
#         for _ in range(self.iterations):
#             ####################
#             #  LINKS TO PATH   #
#             ####################
#             link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")

#             previous_path_state = path_state
#             path_state_sequence, path_state = self.path_update(
#                 link_gather, initial_state=path_state
#             )
#             # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
#             path_state_sequence = tf.concat(
#                 [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
#             )
            
#             ###################
#             #   PATH TO LINK  #
#             ###################
#             path_gather = tf.gather_nd(
#                 path_state_sequence, path_to_link, name="PathToLink"
#             )
            
#             attention_score = self.attention(path_gather)
#             normalized_score = K.softmax(attention_score)
#             weighted_score = normalized_score * path_gather
#             path_gather_score = tf.reduce_sum(weighted_score, axis=1)
            
#             #path_sum = tf.math.reduce_sum(path_gather, axis=1)
            
#             link_state, _ = self.link_update(path_gather_score, states=link_state)


#         ################
#         #  READOUT     #
#         ################

#         occupancy = self.readout_path(path_state_sequence[:, 1:])
#         capacity_gather = tf.gather(link_capacity, link_to_path)
        
#         packets_gather = tf.gather(flow_packets, path_to_link[:, :, 0])
        
#         #queue_delay = occupancy / tf.math.reduce_sum(packets_gather)
#         queue_delay = occupancy / capacity_gather
#         queue_delay = tf.math.reduce_sum(queue_delay, axis=1)

#         #trans_delay = flow_pkt_size_normal / tf.math.reduce_sum(capacity_gather)
#         #trans_delay = flow_pkt_size_normal / tf.math.reduce_sum(capacity_gather)
#         #trans_delay = tf.math.reduce_sum(trans_delay, axis=1)   
        
#         return queue_delay
#     def set_mean_std_scores(self, override_mean_std_scores):
#         assert (
#             type(override_mean_std_scores) == dict
#             and all(kk in override_mean_std_scores for kk in self.mean_std_scores_fields)
#             and all(len(val) == 2 for val in override_mean_std_scores.values())
#         ), "overriden mean-std dict is not valid!"
#         self.mean_std_scores = override_mean_std_scores



class Baseline_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
        "flow_on_rate",
        "flow_ibg"
    }

    name = "Baseline_mb_new"

    def __init__(self, override_min_max_scores=None, name=None):
        super(Baseline_mb, self).__init__()

        self.iterations = 8
        self.path_state_dim = 64
        self.link_state_dim = 64

        if override_min_max_scores is not None:
            self.set_min_max_scores(override_min_max_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            # tf.keras.layers.LSTMCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )
        #self.link_update = tf.keras.layers.GRUCell(
        #    self.link_state_dim, name="LinkUpdate"
        # )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=5),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="PathEmbedding",
        )
        self.attention = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
            tf.keras.layers.Dense(
                self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
            ),
            ]
        )
        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=2),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.link_state_dim // 2, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
            ],
            name="PathReadout",
        )

    def set_min_max_scores(self, override_min_max_scores):
        assert (
            type(override_min_max_scores) == dict
            and all(kk in override_min_max_scores for kk in self.min_max_scores_fields)
            and all(len(val) == 2 for val in override_min_max_scores.values())
        ), "overriden min-max dict is not valid!"
        self.min_max_scores = override_min_max_scores

    @tf.function
    def call(self, inputs):
        # Ensure that the min-max scores are set
        assert self.min_max_scores is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"] 
        flow_packets = inputs["flow_packets"] 
        flow_packet_size = inputs["flow_packet_size"] 
        link_capacity = inputs["link_capacity"] 
        link_to_path = inputs["link_to_path"] 
        path_to_link = inputs["path_to_link"] 

        flow_ibg=inputs["flow_ibg"]
        flow_on_rate= inputs["flow_on_rate"]
        

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.min_max_scores["flow_traffic"][0])
                    * self.min_max_scores["flow_traffic"][1],
                    (flow_packets - self.min_max_scores["flow_packets"][0])
                    * self.min_max_scores["flow_packets"][1],
                    (flow_packet_size - self.min_max_scores["flow_packet_size"][0])
                    * self.min_max_scores["flow_packet_size"][1],
                    (flow_ibg - self.min_max_scores["flow_ibg"][0])
                    * self.min_max_scores["flow_ibg"][1],
                    (flow_on_rate-self.min_max_scores["flow_on_rate"][0])
                    * self.min_max_scores["flow_on_rate"][1],
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for links
        link_state = self.link_embedding(
            tf.concat(
                [
                    (link_capacity - self.min_max_scores["link_capacity"][0])
                    * self.min_max_scores["link_capacity"][1],
                    load,
                ],
                axis=1,
            ),
        )

        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update(
                link_gather, initial_state= path_state
            )
            # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #   PATH TO LINK  #
            ###################
            path_gather = tf.gather_nd(
                path_state_sequence, path_to_link, name="PathToRLink"
            )

            attention_coef = self.attention(path_gather)
            normalized_score = K.softmax(attention_coef)
            weighted_score = normalized_score * path_gather
            
            path_gather_score = tf.math.reduce_sum(weighted_score, axis=1)
           
            link_state, _ = self.link_update(path_gather_score, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        flow_pkt_size = flow_packet_size / 1e9
        flow_pkt_size = tf.expand_dims(flow_pkt_size,axis=1)
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay_sequence = occupancy / capacity_gather
        # transmission_delay_sequence = flow_pkt_size / capacity_gather
        queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        # transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        
        return queue_delay