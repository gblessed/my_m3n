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
import numpy as np
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
      super().__init__()
      self.d_model = d_model
      # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
      self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    # def compute_mask(self, *args, **kwargs):
    #   return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
      length = tf.shape(x)[1]
    
      # This factor sets the relative scale of the embedding and positonal_encoding.
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x = x + self.pos_encoding[tf.newaxis, :length, :]
      return x
    
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x , use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
              dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  

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

        # self.attention = tf.keras.Sequential(
        #     [tf.keras.layers.Input(shape=(None, None, self.path_state_dim)),
        #     tf.keras.layers.Dense(
        #         self.path_state_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01)    
        #     ),
        #     ]
        # )

            # GRU Cells used in the Message Passing step
        # self.path_update = tf.keras.layers.RNN(
        #     tf.keras.layers.GRUCell(self.path_state_dim, name="PathUpdate",
        #     ),
        #     return_sequences=True,
        #     return_state=True,
        #     name="PathUpdateRNN",
        # # )
        # self.link_update = tf.keras.layers.GRUCell(
        #     self.link_state_dim, name="LinkUpdate",
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
    
    

        self.sample_encoder = Encoder(num_layers=4,
                        d_model=self.path_state_dim,
                        num_heads=8,
                        dff=13)
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

        # print("path_to_link shape", path_to_link.shape, path_to_link[0])
        # print("link_to_path shape", link_to_path.shape, link_to_path[0])

        linkG= tf.gather(link_state, link_to_path, name="LinkToPath")
        # original_lengths = tf.ragged.row_lengths(linkG)
      
        linkG  = linkG.to_tensor()
        
        Pnlink = tf.concat([tf.expand_dims(path_state, axis=1), linkG], axis=1)
       
        # print("positional encoding", embed_pt(Pnlink).shape)



        path_state_sequence = self.sample_encoder(Pnlink, training=True)  
        path_state_sequence = tf.RaggedTensor.from_tensor(path_state_sequence, lengths=flow_length, )
        # print("path_state_sequence", path_state_sequence.shape)
 
        # Iterate t times doing the message passing

        # for _ in range(self.iterations):
        #     ####################
        #     #  LINKS TO PATH   #
        #     ####################
            
        #     link_gather = tf.gather(link_state, link_to_path, name="LinkToPath").to_tensor()
            
        #     link_gather =tf.concat([tf.expand_dims(path_state, axis=1), link_gather], axis=1)

        #     # previous_path_state = path_state
        #     # path_state_sequence, path_state = self.path_update(
        #     #     link_gather, initial_state=path_state
        #     # )
        #     path_state_sequence  = sample_encoder(Pnlink, training=t)
        #     # We select the element in path_state_sequence so that it corresponds to the state before the link was considered
        #     path_state_sequence = tf.concat(
        #         [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
        #     )
            
        #     ###################
        #     #   PATH TO LINK  #
        #     ###################
        #     path_gather = tf.gather_nd(
        #         path_state_sequence, path_to_link, name="PathToLink"
        #     )
            
        #     attention_coef = self.attention(path_gather)
        #     normalized_score = K.softmax(attention_coef)
        #     weighted_score = normalized_score * path_gather
            
        #     path_gather_score = tf.math.reduce_sum(weighted_score, axis=1)
            
        #     link_state, _ = self.link_update(path_gather_score, states=link_state)

        ################
        #  READOUT     #
        ################

        occupancy = self.readout_path(path_state_sequence[:, :])

            
    
        capacity_gather = tf.gather(link_capacity, link_to_path)
        
        # print("occupancy", occupancy.shape, capacity_gather.shape )
        # print("divide", (occupancy[0],  capacity_gather[0]) )
        queue_delay = occupancy / capacity_gather
        queue_delay = tf.math.reduce_sum(queue_delay, axis=1)

        return queue_delay
    
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

        return {m.name: m.result() for m in self.metrics}
    
    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker]
    
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
       
