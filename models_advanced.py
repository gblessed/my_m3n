"""
These models are used to perform the training and 
obtain the parameters saved in the ckpk folder.
"""

import tensorflow as tf
import numpy as np
import keras.backend as K
from tfkan.layers import DenseKAN
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
        # Ensure that the min-max scores are set
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
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay_sequence = occupancy / capacity_gather
        # transmission_delay_sequence = flow_pkt_size / capacity_gather

        queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        # transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        return queue_delay 
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
                tf.keras.layers.Input(shape=8),
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
        # Ensure that the min-max scores are set
        assert self.mean_std_scores_fields is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        flow_traffic = inputs["flow_traffic"] 
        flow_packets = inputs["flow_packets"] 
        flow_packet_size = inputs["flow_packet_size"] 
        link_capacity = inputs["link_capacity"] 
        link_to_path = inputs["link_to_path"] 
        path_to_link = inputs["path_to_link"] 

        flow_ibg=inputs["ibg"]
        flow_on_rate= inputs["flow_on_rate"]
        

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.mean_std_scores["flow_traffic"][0])
                    * self.mean_std_scores["flow_traffic"][1],
                    (flow_packets - self.mean_std_scores["flow_packets"][0])
                    * self.mean_std_scores["flow_packets"][1],
                    (flow_packet_size - self.mean_std_scores["flow_packet_size"][0])
                    * self.mean_std_scores["flow_packet_size"][1],
                    (flow_ibg - self.mean_std_scores["flow_ibg"][0])
                    * self.mean_std_scores["flow_ibg"][1],
                    (flow_on_rate-self.mean_std_scores["flow_on_rate"][0])
                    * self.mean_std_scores["flow_on_rate"][1],
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
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay_sequence = occupancy / capacity_gather
        # transmission_delay_sequence = flow_pkt_size / capacity_gather
        queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        # transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        
        return queue_delay
        # return queue_delay + transmission_delay





class Baseline_mb_attn(tf.keras.Model):
    mean_std_scores_fields = {
        "flow_traffic",
        "flow_packets",
        ##new
        "flow_pkts_per_burst",
        "flow_bitrate_per_burst",
        "flow_p90PktSize",
        "link_capacity",

        "flow_packet_size",
        "link_capacity",
        "flow_on_rate",
        "flow_ibg"
    }

    name = "Baseline_mb_attn_node"
    

    def __init__(self, override_mean_std_scores=None, name=None):
        super(Baseline_mb_attn, self).__init__()
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

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=5 + 4),
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
                tf.keras.layers.Input(shape=2+2),
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

        # self.readout_path = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=(None, self.path_state_dim)),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 2, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 4, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
        #         tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus)
        #     ],
        #     name="PathReadout",
        # )
        self.readout_path = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            DenseKAN(int(self.link_state_dim // 2)),
            # DenseKAN(int(self.path_state_dim / 4)),
            DenseKAN(1),

        ])
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
        # Ensure that the min-max scores are set
        assert self.mean_std_scores_fields is not None, "the model cannot be called before setting the min-max scores!"

        # Process raw inputs
        # print("inputs", inputs.keys())
        flow_traffic = inputs["flow_traffic"] 
        flow_packets = inputs["flow_packets"] 
        flow_packet_size = inputs["flow_packet_size"] 
        link_capacity = inputs["link_capacity"] 
        link_to_path = inputs["link_to_path"] 
        path_to_link = inputs["path_to_link"] 

        flow_ibg=inputs["flow_ibg"]
        flow_on_rate= inputs["flow_on_rate"]

        flow_p90pktsize = inputs["flow_p90PktSize"]
        flow_pkt_per_burst = inputs["flow_pkts_per_burst"]
        flow_bitrate = inputs["flow_bitrate_per_burst"]
        flow_length = inputs["flow_length"]
        max_link_load = inputs["max_link_load"]
        devices_to_link  = inputs["devices_to_link"] #contains the ids of links connected to each device ##[[0, 1], [2, 3, 4, 5], [8, 9, 6, 7], [10, 11, 12], 


        link_to_dev =  devices_to_link *0 +  tf.expand_dims(tf.range(tf.shape(devices_to_link)[0]),1)

        node_degrees = tf.math.reduce_sum(tf.ones_like(devices_to_link), axis=1) 
        links_node_degs = tf.reshape(tf.gather(node_degrees, link_to_dev, name="LinkToPath"), [-1])/ tf.math.reduce_sum(tf.gather(node_degrees, link_to_dev, name="LinkToPath"))
    
        

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
        normal_load = tf.math.divide(load, tf.squeeze(max_link_load))
        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    (flow_traffic - self.mean_std_scores["flow_traffic"][0])
                    * self.mean_std_scores["flow_traffic"][1],
                    (flow_packets - self.mean_std_scores["flow_packets"][0])
                    * self.mean_std_scores["flow_packets"][1],
                    (flow_packet_size - self.mean_std_scores["flow_packet_size"][0])
                    * self.mean_std_scores["flow_packet_size"][1],
                    (flow_ibg - self.mean_std_scores["flow_ibg"][0])
                    * self.mean_std_scores["flow_ibg"][1],
                    (flow_on_rate-self.mean_std_scores["flow_on_rate"][0])
                    * self.mean_std_scores["flow_on_rate"][1],
                    
                    (flow_p90pktsize - self.mean_std_scores["flow_p90PktSize"][0])
                    * self.mean_std_scores["flow_p90PktSize"][1],
                     (flow_bitrate - self.mean_std_scores["flow_bitrate_per_burst"][0])
                    * self.mean_std_scores["flow_bitrate_per_burst"][1],
                    (flow_pkt_per_burst - self.mean_std_scores["flow_pkts_per_burst"][0])
                    * self.mean_std_scores["flow_pkts_per_burst"][1],
                     tf.expand_dims(tf.cast(flow_length, dtype=tf.float32), 1),
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
                    tf.expand_dims(tf.cast(links_node_degs, tf.float32),1)
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

        # occupancy = self.readout_path(path_state_sequence[:, 1:])
        # flow_pkt_size = flow_packet_size / 1e9
        # flow_pkt_size = tf.expand_dims(flow_pkt_size,axis=1)
        # capacity_gather = tf.gather(link_capacity, link_to_path)
        
        # queue_delay_sequence = occupancy / capacity_gather
        # # transmission_delay_sequence = flow_pkt_size / capacity_gather
        # queue_delay = tf.math.reduce_sum(queue_delay_sequence, axis=1)
        # # transmission_delay = tf.math.reduce_sum(transmission_delay_sequence, axis=1)
        
        # return queue_delay
        # return queue_delay + transmission_delay
        capacity_gather = tf.gather(link_capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(flow_length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather,
                                         axis=1)


        return queue_delay 




class Baseline_cbr_mb_std(tf.keras.Model):
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

    # name = "Baseline_cbr_mb_std"
    # print("name->",name )
    def __init__(self, override_mean_std_scores=None, name=None):
        super(Baseline_cbr_mb_std, self).__init__()

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


        # self.device_link_attention = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
        #     ),
        #     ]
        # )


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
        # self.device_update = tf.keras.layers.GRUCell(
        #     self.link_state_dim, name="DeviceUpdate",
        # )
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
                tf.keras.layers.Input(shape=4),
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

        # self.device_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=4),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             )
        #     ],
        #     name="DeviceEmbedding",
        # )
        # self.readout_path = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=(None, self.path_state_dim)),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 2, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 4, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
        #         tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus)
        #     ],
        #     name="PathReadout",
        # )
        self.readout_path = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            DenseKAN(int(self.link_state_dim // 1)),
            # DenseKAN(int(self.path_state_dim / 4)),
            DenseKAN(1),

        ])    
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
        # print("inputs_cbr", inputs.keys())
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
        devices_to_link  = inputs["devices_to_link"] #contains the ids of links connected to each device ##[[0, 1], [2, 3, 4, 5], [8, 9, 6, 7], [10, 11, 12], 
        flow_ipg_var = inputs["flow_ipg_var"]
        link_capacity = inputs["link_capacity"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]
        device_to_link = inputs["devices_to_link"]
        device_to_path = inputs["devices_to_path"]

        
        # print("adj", inputs["adj_matrix"])
        link_to_dev =  devices_to_link *0 +  tf.expand_dims(tf.range(tf.shape(devices_to_link)[0]),1)

        # print("devices_to_link", devices_to_link ) ##    [[0,0 ,0]
        node_degrees = tf.math.reduce_sum(tf.ones_like(devices_to_link), axis=1) 
        # print("sh",  (tf.gather(node_degrees, link_to_dev, name="LinkToPath") ,axis  =0) )
        links_node_degs = tf.reshape(tf.gather(node_degrees, link_to_dev, name="LinkToPath"), [-1])/ tf.math.reduce_sum(tf.gather(node_degrees, link_to_dev, name="LinkToPath"))
        # links_node_degs =  tf.concat(tf.gather(node_degrees, link_to_dev, name="LinkToPath"), axis=0)/ tf.math.reduce_sum(tf.gather(node_degrees, link_to_dev, name="LinkToPath"))

 

        # print("len_devices_to_link", devices_to_link.shape)



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
                    tf.expand_dims(tf.cast(links_node_degs, tf.float32),1)
                ],
                axis=1,
            ),
        )

        device_init_feats  =  tf.concat(
                [
                   (link_capacity - self.mean_std_scores["link_capacity"][0])
                    * self.mean_std_scores["link_capacity"][1],
                    load,
                    normal_load,
                    tf.expand_dims(tf.cast(links_node_degs, tf.float32),1)
                ],
                axis=1,
            )
        # device_gather = tf.gather(link_state, devices_to_link, name="LinkToPath")

        # device_feats = tf.math.reduce_sum(tf.gather(device_init_feats, device_to_link, name="DeviceToPath"), axis=1)
        # device_states = self.device_embedding(device_feats)


        # device_links = tf.gather(link_state, device_to_link, name="DeviceToPath"), 
        # device_links_raw = self.device_link_attention(device_links) #+ tf.broadcast_to(self.device_attention(device_states), device_links.shape) 

        # print(device_links_raw.shape, self.device_attention(device_states).shape)
        # Iterate t times doing the message passing
        for _ in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            # device_path_gather = tf.gather(device_states, device_to_path, name="DeviceToPath")
            # link_gather = tf.math.sigmoid(device_path_gather) * link_gather
           


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


            # device_link_gather = tf.gather(link_state, device_to_link, name="DeviceToPath")
            # attention_coef = self.device_link_attention(device_link_gather)
            # normalized_score = K.softmax(attention_coef)
            # weighted_score = normalized_score * device_link_gather
            # device_gather_score = tf.math.reduce_sum(weighted_score, axis=1)
            # # print("device_links_gather",device_link_gather.shape)

            # device_states, _ = self.device_update(device_gather_score, states=device_states)




        ################
        #  READOUT     #
        ################

        # occupancy = self.readout_path(path_state_sequence[:, 1:])

        # capacity_gather = tf.gather(link_capacity, link_to_path)
        
        # queue_delay = occupancy / capacity_gather
        # queue_delay = tf.math.reduce_sum(queue_delay, axis=1)

        # return queue_delay
        capacity_gather = tf.gather(link_capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(flow_length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather,
                                         axis=1)


        return queue_delay 
