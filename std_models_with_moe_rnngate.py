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

        initial_path_state = path_state

        # initial_original_state = link_state           
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
            

            # link_state +=initial_original_state
            previous_path_state +=initial_path_state
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

        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        queue_delay = occupancy / capacity_gather
        queue_delay = tf.math.reduce_sum(queue_delay, axis=1)

        return queue_delay
    
class Baseline_cbr_MoE(tf.keras.Model):
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

    name = "ufpa_ericsson_cbr_mb_moe"

    def __init__(self, override_mean_std_scores=None, name=None):
        super(Baseline_cbr_MoE, self).__init__()
        self.expert1 = Baseline_cbr_mb(name = "expert1")
        self.expert2 = Baseline_cbr_mb(name = "expert2")
        self.expert3 = Baseline_cbr_mb(name = "expert3")
        self.expert4 = Baseline_cbr_mb(name = "expert4")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delay_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="delay_mape")
        self.jitter_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="jitter_mape")
        self.pkts_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="pkts_metric")
        self.iterations = 12
        self.path_state_dim = 16
        self.link_state_dim = 16
        self.gate = tf.keras.Sequential(
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

        
            tf.keras.layers.Dense(4, activation=tf.keras.activations.sigmoid)
        ],
        name="Gate",
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

        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate",
            ),
            return_sequences=True,
            return_state=True,
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
            ]
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
        
        link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")

        path_state_sequence, path_state = self.path_update(
            link_gather, initial_state=path_state
        )

        route = self.gate(path_state)








        # print(route[:10])
        # flow_1_indices = tf.where(route > 0.5)[:, 0]
        # flow_2_indices = tf.where(route <= 0.5)[:, 0]
        # print(route)
        # print("flow1", flow_1_indices)
        # print("flow2", flow_2_indices)
        # print("flow2", path_to_link[0])

        # values = flow_1_indices
        # tensor = tf.cast(path_to_link[:,:, 0].to_tensor(), tf.int64)
        # print("vals", values.shape)
        # print("tensor", tensor[0])
        

        # print(tf.sets.difference(tensor, tf.expand_dims(values, axis=1)).values)
        # print(tf.sets.difference(tf.expand_dims(path_to_link[:,:, 0], axis=0), tf.expand_dims(values, axis=0)).values)



        # mask1 = tf.round(route)
        # mask2 = tf.round(1- route)



        mask1 = binary_activation(route[:, 0])
        mask2 = binary_activation(route[:, 1])
        mask3 = binary_activation(route[:, 2])
        mask4 = binary_activation(route[:, 3])
        # print("route[:, 0]", route[:, 0][:5])


    
        # exp2 = tf.where(route <= 0.7)
        # mask2 = binary_activation2(route)

        self.expert1.mean_std_scores = self.mean_std_scores
        delay1 = self.expert1(inputs)
        self.expert2.mean_std_scores = self.mean_std_scores
        delay2 = self.expert2(inputs)
        self.expert3.mean_std_scores = self.mean_std_scores
        delay3 = self.expert3(inputs)
        self.expert4.mean_std_scores = self.mean_std_scores
        delay4 = self.expert4(inputs)
        # print("exp1", tf.reduce_sum(mask1), "exp2", tf.reduce_sum(mask2))
        # delays = delay1 * tf.cast(mask1, tf.float32) * route + delay2 * tf.cast(mask2, tf.float32) * route
        # print(delay1.shape, mask1.shape, route[:, 0].shape)
        # print("delay1 * tf.cast(mask1, tf.float32) * route[:, 0]", (delay1[:, 0] * tf.cast(mask1, tf.float32) * route[:, 0])[:5])
        delays = delay1[:, 0] * tf.cast(mask1, tf.float32) * route[:, 0] + delay2[:, 0] * tf.cast(mask2, tf.float32) * route[:, 1]
        delays += delay3[:, 0] * tf.cast(mask3, tf.float32) * route[:, 2] +    delay4[:, 0] * tf.cast(mask4, tf.float32) * route[:, 3] 

        # print(delays.shape,delay1.shape, route.shape, mask1.shape)


        ## soft approach

        # self.expert1.mean_std_scores = self.mean_std_scores
        # delay1 = self.expert1(inputs)
        # self.expert2.mean_std_scores = self.mean_std_scores
        # delay2 = self.expert2(inputs)
        # delays = delay1 * route[:, 0] + delay2 * route[:, 1]

        # delays = delay1 * tf.cast(mask1, tf.float32) + delay2 * tf.cast(mask2, tf.float32)

        # print(delays)
        # delays = tf.zeros_like(route)
        # self.expert1.mean_std_scores = self.mean_std_scores
        # delay1 = self.expert1(inputs)
      

        # indices = tf.cast(exp1[:, 0], tf.int64)  # A list of coordinates to update.
        # delay1 = tf.gather(delay1,indices)
        # indices = tf.expand_dims(indices,1)
        # delays = tf.tensor_scatter_nd_update(delays, indices, delay1)

        # self.expert2.mean_std_scores = self.mean_std_scores
        # delay1 = self.expert2(inputs)
        # indices = tf.cast(exp2[:, 0], tf.int64)  # A list of coordinates to update.

        # delay1 = tf.gather(delay1,indices)
        # indices = tf.expand_dims(indices,1)
        # delays = tf.tensor_scatter_nd_update(delays, indices, delay1)

        # print(delays)
        
   
 

        return delays
    


    def train_step(self, data):
        x, delay, jitter, pktsdropped = data
        # print("packets dropped ",pktsdropped[:10, ] , tf.squeeze(x["flow_traffic"], axis=1)[:10])
        # pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        # print("vals===", pktsdropped)
        with tf.GradientTape() as tape:
            delay_pred = self(x, training=True)  # Forward pass
            # Compute our own lossMeanSquaredLogarithmicError
            # jitter_pred = delay_pred
            # pkts_pred  = delay_pred
            # mean_squared_logarithmic_error
            loss = tf.keras.losses.mean_absolute_percentage_error(delay, delay_pred)
            # loss+= .33* tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
            # loss += .33* tf.keras.losses.mean_absolute_percentage_error(pktsdropped, pkts_pred)
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
        # self.delay_metric.update_state(delay, delay_pred)
        # self.jitter_metric.update_state(jitter, jitter_pred)
        # self.pkts_metric.update_state(pktsdropped, pkts_pred)

        return {"loss": self.loss_tracker.result()}
        return {"loss": self.loss_tracker.result()}
    
        # return {"loss": self.loss_tracker.result(), "train_delay_mape": self.delay_metric.result(), "train_jitter_mape": self.jitter_metric.result(), "train_pkts_mape": self.pkts_metric.result(),}
    

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker,  ]
        return [self.loss_tracker, self.delay_metric, ]
    

    def compute_loss(self, delay, delay_pred,sample_weight= None):

        loss = tf.keras.losses.mean_absolute_percentage_error(delay, delay_pred)
        # loss+= .33* tf.keras.losses.mean_absolute_percentage_error(jitter, jitter_pred)
        # loss += .33* tf.keras.losses.mean_absolute_percentage_error(pktsdropped, pkts_pred)
        self.loss_tracker.update_state(loss)
        ## increase the complexity of the model
        ## try to use sigmoid for the packets dropped
        return loss

    def test_step(self, data):
        # Unpack the data
        x, delay, jitter, pktsdropped  = data
        # pktsdropped = pktsdropped/tf.squeeze(x["flow_traffic"], axis=1)
        performance_metrics = [delay, jitter, pktsdropped]
        # Compute predictions
        delay_pred = self(x, training=False)
        predictions = [delay_pred ]

        # Updates the metrics tracking the loss
        self.compute_loss(delay, delay_pred)

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
        


def binary_activation(x):

    cond = tf.less_equal(x, tf.ones(tf.shape(x))*0.5)
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

def binary_activation2(x):

    cond = tf.greater(x, tf.ones(tf.shape(x))*0.5)
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

