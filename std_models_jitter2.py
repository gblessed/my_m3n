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
import tensorflow_addons as tfa



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
        # self.jitter_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="jitter_mape")
        # # self.pkts_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="pkts_metric")

        self.iterations = 12 
        self.path_state_dim = 16
        self.link_state_dim = 16 
        # self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # self.mae_metric = tf.keras.metrics.BinaryAccuracy(name="acc")

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

        # self.attention2 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
        #     ),
        #     ]
        # )        
        # self.attention3 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
        #     ),
        #     ]
        # )        
        # self.attention4 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
        #     ),
        #     ]
        # )



        # self.a_T = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )


        # self.a_T = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )

        # self.a_T2 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )

        # self.a_T3 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )

        # self.a_T4 = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )

        # self.project_back = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, 4*self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation='linear'   
        #     ),
        #     ]
        # )
            # GRU Cells used in the Message Passing step
        self.path_update =  tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim,name="PathUpdate",
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
                    ),
    
                
            ],
            name="PathEmbedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3 + 0),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
               
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.selu,
                    kernel_initializer='lecun_uniform',
                    ),
                
                
            ],
            name="LinkEmbedding",
        )
        # self.layer_norm =tf.keras.layers.LayerNormalization()
        # self.layer_norm2 =tf.keras.layers.LayerNormalization()



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

        # self.projection = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=self.path_state_dim),
        #         tf.keras.layers.Dense(
        #             self.path_state_dim, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
               
        #         tf.keras.layers.Dense(
        #             self.path_state_dim, activation=tf.keras.activations.selu,
        #             kernel_initializer='lecun_uniform',
        #             ),
    
                
        #     ],
    
        # )
        # self.jitter_readout_path = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=(None, self.path_state_dim)),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 2, activation=tf.keras.activations.gelu
        #         ),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 4, activation=tf.keras.activations.gelu
        #         ),
        #         tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus),
        #     ],
        #     name="JitterReadout",
        # )

        # self.pkt_loss_readout_path = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Input(shape=(None, self.path_state_dim)),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 2, activation=tf.keras.activations.gelu
        #         ),
        #         tf.keras.layers.Dense(
        #             self.link_state_dim // 4, activation=tf.keras.activations.gelu
        #         ),
        #         tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
        #     ],
        #     name="pkt_lossReadout",
        # )    
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
        link_capacity_and_node_type = inputs["link_capacity_and_node_type"]     
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]
        # greater_indices = inputs["greater_indices"]

        # print(flow_traffic.shape)

        # x = tf.concat(
        #     [
        #         (  tf.expand_dims(link_capacity_and_node_type[:, 0], 1)- self.mean_std_scores["link_capacity"][0])
        #         * self.mean_std_scores["link_capacity"][1],
        #         tf.expand_dims(link_capacity_and_node_type[:, 1], 1),
        #         # tf.expand_dims(link_capacity_and_node_type[:, 2], 1),
        #         # tf.expand_dims(link_capacity_and_node_type[:, 3], 1),
        #         # tf.expand_dims(link_capacity_and_node_type[:, 4], 1),
        #         # tf.expand_dims(link_capacity_and_node_type[:, 5], 1),
        #         # tf.expand_dims(link_capacity_and_node_type[:, 6], 1),
        #     ],
        #     axis=1,
        # ),
        
        # link_gather = tf.gather(x, link_to_path, name="LinkToPath")
        # print("linkGather", link_gather[0] )

        flow_pkt_size_normal = (flow_packet_size - self.mean_std_scores["flow_packet_size"][0]) \
                    * self.mean_std_scores["flow_packet_size"][1],

        path_gather_traffic = tf.gather(flow_traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / (link_capacity * 1e9)
        normal_load = tf.math.divide(load, tf.squeeze(max_link_load))
        # print("load", load.shape,    (  tf.expand_dims(link_capacity_and_node_type[:, 0], 1)- self.mean_std_scores["link_capacity"][0])
        #             * self.mean_std_scores["link_capacity"][1],  ).shape)
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
                   (  tf.expand_dims(link_capacity_and_node_type[:, 0], 1)- self.mean_std_scores["link_capacity"][0])
                    * self.mean_std_scores["link_capacity"][1],
                    load,
                    normal_load,
                    # tf.expand_dims(link_capacity_and_node_type[:, 1], 1),
                    # tf.expand_dims(link_capacity_and_node_type[:, 2], 1),
                    # tf.expand_dims(link_capacity_and_node_type[:, 3], 1),
                    # tf.expand_dims(link_capacity_and_node_type[:, 4], 1),
                    # tf.expand_dims(link_capacity_and_node_type[:, 5], 1),
                    # tf.expand_dims(link_capacity_and_node_type[:, 6], 1),
                ],
                axis=1,
            ),
        )



        initial_path_state = path_state

        initial_original_state = link_state

        # Iterate t times doing the message passing
        for _  in range(self.iterations):
            ####################
            #  LINKS TO PATH   #
            ####################
            # print(tf.norm(initial_path_state - path_state))
            
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            
            # link_gather = tf.concat([link_gather, initial_path_state], axis=2)
            
            # previous_path_state = path_state + initial_path_state
            link_state +=initial_original_state
            previous_path_state = path_state + initial_path_state
           


            path_state_sequence, path_state= self.path_update(
                link_gather, initial_state=path_state
            )
            # path_state_sequence = self.layer_norm(path_state_sequence)
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
            # attention_coef = self.a_T(attention_coef)
            normalized_score = K.softmax(attention_coef)
            # print("normalized_score", normalized_score.shape, normalized_score[0].shape)
            weighted_score = normalized_score * path_gather        
            path_gather_score = tf.math.reduce_sum(weighted_score, axis=1)

            link_state, _ = self.link_update(path_gather_score, states=link_state)
            # link_state = self.layer_norm2(link_state)


        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, 1:])
        # occupancy = all_reads[:,:, 0]
        # pkts_dropped = all_reads[:,:, 1]
        # jitter_occupancy = all_reads[:,:, 1]

        # binaryclass = self.readout_path(path_state)

        # condition = pkts_dropped > 9466
       
        # pkts_dropped = tf.where(condition, flow_packets, pkts_dropped)
      
     
        # jitter_occupancy = self.jitter_readout_path(path_state_sequence[:, 1:])

        capacity_gather = tf.gather(link_capacity, link_to_path)
        delay_sequence = occupancy / capacity_gather
        # jitter_delay_sequence = jitter_occupancy / capacity_gather

        # jitter = tf.math.reduce_sum(jitter_delay_sequence, axis=1)
    
        delay = tf.math.reduce_sum(delay_sequence, axis=1)

        # delay = tf.squeeze(tf.gather(delay, greater_indices), axis = 0)
 

        return delay
    
    def train_step(self, data):
        x,jitter, _,_, = data
        # x, jitter, _, _= data
        # x, jitter = data


        # print("packets dropped ",pktsdropped[:10, ] , tf.squeeze(x["flow_traffic"], axis=1)[:10])
        # pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        # print("vals===", pktsdropped)
        with tf.GradientTape() as tape:
            jitter_pred = self(x, training=True)  # Forward pass
            # Compute our own lossMeanSquaredLogarithmicError
            # mean_squared_logarithmic_error
            # loss = .33 * tf.keras.losses.mean_squared_logarithmic_error(delay, delay_pred)
            loss=  self.compute_loss(y=jitter, y_pred=jitter_pred)
            # loss += .33* tf.keras.losses.mean_squared_logarithmic_error(pktsdropped, pkts_pred)

        # loss *=.3
        # loss = (delayloss + jitterloss + pktsloss) / 3
        # Compute gradients
        trainable_vars = self.trainable_variables
        # print("lennnn ", len(trainable_vars))
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(jitter, jitter_pred) 
        # Compute our own metrics
        # self.loss_tracker.update_state(loss)
        # self.delay_metric.update_state(delay, delay_pred)
        # self.jitter_metric.update_state(jitter, jitter_pred)
        # self.pkts_metric.update_state(pktsdropped, pkts_pred)
        # self.loss_tracker.update_state(loss)
        # self.mae_metric.update_state(jitter, jitter_pred)
        # return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
        return {m.name: m.result() for m in self.metrics}
    
    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker, self.delay_metric ]
    
    # def compute_loss(self, jitter, jitter_pred, sample_weight= None):
       
    #     loss =  tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
    #     # loss+= .33* tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
    #     # loss += .33* tf.keras.losses.mean_absolute_percentage_error(pktsdropped, pkts_pred)
    #     self.loss_tracker.update_state(loss)
    #     ## increase the complexity of the model
    #     ## try to use sigmoid for the packets dropped
    #     return loss
    
    def test_step(self, data):
        # Unpack the data
        x, jitter, _, _,  = data
        # x, jitter = data

        # pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        performance_metrics = [jitter]
        # Compute predictions
        jitter_pred= self(x, training=False)
        predictions = [jitter_pred ]
     

        # Updates the metrics tracking the loss
        self.compute_loss(y= jitter, y_pred=jitter_pred)
    
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
       