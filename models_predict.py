"""
These models are used to predict the test dataset.
Unlike the models in modes.py, these models determine the final output delay 
by comparing the transmission delay with the calculated delay.
"""

import tensorflow as tf
import numpy as np
import keras.backend as K
class Baseline_cbr_mb(tf.keras.Model):
    mean_std_scores_fields = {
        # "mean_std_scores_fields",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
        "flow_ipg_mean",
        "flow_ipg_var",
        "flow_on_rate",
        "flow_traffic",
        "flow_pkts_per_burst",
        "flow_bitrate_per_burst",
        "flow_p90PktSize",
  
    }
    mean_std_scores = None 

    name = "Baseline_cbr_mb_new2"
    def __init__(self, override_mean_std_scores=None, name=None):
        super(Baseline_cbr_mb, self).__init__()

        self.iterations = 12
        self.path_state_dim = 16
        self.link_state_dim = 16

        if override_mean_std_scores is not None:
            self.set_mean_std_scores(override_mean_std_scores)
        if name is not None:
            assert type(name) == str, "name must be a string"
            self.name = name
        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate"),
            return_sequences=True,
            return_state=True,
            name="PathUpdateRNN",
        )

        self.attention = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
            tf.keras.layers.Dense(
                self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
            ),
            ]
        )

        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=8+3),
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
                tf.keras.layers.Input(shape=2),
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
                    self.link_state_dim // 2, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(
                    self.link_state_dim // 4, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus)
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
    def set_mean_std_scores(self, override_mean_std_scores):
        assert (
            type(override_mean_std_scores) == dict
            and all(kk in override_mean_std_scores for kk in self.mean_std_scores_fields)
            and all(len(val) == 2 for val in override_mean_std_scores.values())
        ), "overriden mean-std dict is not valid!"
        self.mean_std_scores = override_mean_std_scores

    @tf.function
    def call(self, inputs):
        assert self.mean_std_scores_fields is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"]
        flow_packets = inputs["flow_packets"]
        flow_packet_size = inputs["flow_packet_size"]
        flow_type = inputs["flow_type"]
        flow_p90pktsize = inputs["flow_p90PktSize"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]
        flow_bitrate = inputs["flow_bitrate_per_burst"]
        flow_pkt_per_burst = inputs["flow_pkts_per_burst"]
        # Add new features
        flow_ipg_mean = inputs["flow_ipg_mean"] 
        flow_ipg_var = inputs["flow_ipg_var"]
        flow_on_rate= inputs["flow_on_rate"]
        # Add new features
        
        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.flow_embedding(
            tf.concat(
                [
                    (flow_traffic - self.mean_std_scores["flow_traffic"][0])
                    * self.mean_std_scores["flow_traffic"][1],
                    (flow_packets - self.mean_std_scores["flow_packets"][0])
                    * self.mean_std_scores["flow_packets"][1],
                    (flow_packet_size - self.mean_std_scores["flow_packet_size"][0])
                    * self.mean_std_scores["flow_packet_size"][1],
                    (flow_ipg_mean - self.mean_std_scores["flow_ipg_mean"][0])
                    * self.mean_std_scores["flow_ipg_mean"][1],
                    (flow_ipg_var - self.mean_std_scores["flow_ipg_var"][0])
                    * self.mean_std_scores["flow_ipg_var"][1],
                    (flow_on_rate - self.mean_std_scores["flow_on_rate"][0])
                    * self.mean_std_scores["flow_on_rate"][1],
                    flow_type,
                     (flow_p90pktsize - self.mean_std_scores["flow_p90PktSize"][0])
                    * self.mean_std_scores["flow_p90PktSize"][1],
                    (flow_pkt_per_burst - self.mean_std_scores["flow_pkts_per_burst"][0])
                    * self.mean_std_scores["flow_pkts_per_burst"][1],
                    (flow_bitrate - self.mean_std_scores["flow_bitrate_per_burst"][0])
                    * self.mean_std_scores["flow_bitrate_per_burst"][1],
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
        # capacity_gather表示path上的link的容量，表示为向量
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay_sequence = occupancy / capacity_gather
        transmission_delay_sequence = flow_pkt_size / capacity_gather
        # delay_sequence表示path在各个link上的delay
        queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        transmission_delay = transmission_delay * 1e3
        # 如果queue_delay中预测的值要小于transmission_delay的值，即某个流的延迟，则使用transmission_delay中的值进行覆盖
        condition = queue_delay < transmission_delay
        delay = tf.where(condition, transmission_delay, queue_delay)
        return delay 

class Baseline_mb(tf.keras.Model):
    min_max_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_packet_size",
        "link_capacity",
        "flow_on_rate",
        "flow_ibg",
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

        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )

        # path与link编码
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
        
        # Add new features
        flow_ibg=inputs["flow_ibg"]
        flow_on_rate= inputs["flow_on_rate"]
        # Add new features

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
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            link_state, _ = self.link_update(path_sum, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        flow_pkt_size = flow_packet_size / 1e9
        flow_pkt_size = tf.expand_dims(flow_pkt_size,axis=1)
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay_sequence = occupancy / capacity_gather
        transmission_delay_sequence = flow_pkt_size / capacity_gather

        queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        transmission_delay = transmission_delay * 1e3
        
        condition = queue_delay < transmission_delay
        delay = tf.where(condition, transmission_delay, queue_delay)
        return delay 