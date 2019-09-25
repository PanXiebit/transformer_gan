from model.transformer_9 import Transformer
import tensorflow as tf
from model.base import ModeKeys
from model import model_utils
from utils import metrics
from utils.tokenizer import EOS_ID, PAD_ID


class Generator(Transformer):
    def __init__(self, params, is_train, name_scope, mode=None):
        super(Generator, self).__init__(params, is_train, mode=mode, scope=name_scope)
        self.name_scope = name_scope

    def build_padding_rollout_generator(self, real_inputs, gen_samples, max_len, given_num):
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            if ModeKeys.is_predict_one(self.mode):
                self.attention_bias = None
            else:
                self.attention_bias = model_utils.get_padding_bias(real_inputs)
            self.encoder_outputs = self.encode(real_inputs, self.attention_bias)

            def condition(given_num, _):
                return given_num < max_len

            def inner_loop(given_num, given_y):
                logits = self.decode(given_y, self.encoder_outputs, self.attention_bias)
                next_logits = logits[:, given_num, :]  # [batch, decoder_vocab_size]
                next_probs = tf.nn.softmax(next_logits)
                log_probs = tf.log(next_probs)
                next_sample = tf.multinomial(log_probs, num_samples=1)
                next_sample = tf.cast(next_sample, dtype=tf.int32)
                given_y = tf.concat([given_y[:, :given_num], next_sample], axis=1)
                given_y = tf.pad(given_y, [[0, 0], [0, max_len - given_num - 1]])
                return given_num + 1, given_y

            given_y = gen_samples[:, :given_num]
            init_given_y = tf.pad(given_y, [[0, 0], [0, max_len - given_num]])
            init_given_num = tf.constant(given_num)

            given_num, roll_sample = tf.while_loop(
                cond=condition,
                body=inner_loop,
                loop_vars=[init_given_num, init_given_y],
                shape_invariants=[init_given_num.get_shape(),
                                  tf.TensorShape([None, None])]
            )
            return roll_sample

    def get_reward(self, real_inputs, real_targets, gen_targets, roll_num, discriminator, maxlen=25):
        real_loss = discriminator.get_loss(real_targets, real_inputs)  # [batch ,1]
        base_f_loss = discriminator.get_loss(gen_targets, real_inputs)
        base_reward = 1 / tf.maximum(base_f_loss / real_loss, 1)

        y_sample_mask = tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.float32)
        rewards = []
        roll_losses = []
        for i in range(roll_num):
            for given_num in range(1, maxlen):
                tf.logging.info("roll_num: {}".format(i))
                roll_sample = self.build_padding_rollout_generator(
                    real_inputs=real_inputs,
                    gen_samples=gen_targets,
                    max_len=maxlen,
                    given_num=given_num)
                roll_loss = discriminator.get_loss(
                    gen_targets=roll_sample,
                    real_inputs=real_inputs)  # [batch ,1]
                roll_losses.append(roll_loss)
                cur_reward = tf.maximum(1 / tf.maximum(roll_loss / real_loss, 1) - base_reward, 0.0)
                if i == 0:
                    rewards.append(cur_reward)  # list, [batch,1] * max_len
                else:
                    rewards[given_num-1] += cur_reward

            roll_loss = discriminator.get_loss(gen_targets=gen_targets,
                                                 real_inputs=real_inputs)
            last_reward = tf.maximum(1 / tf.maximum(roll_loss / real_loss, 1) - base_reward, 0.0)
            if i == 0:
                rewards.append(last_reward)
            else:
                rewards[maxlen -1] += last_reward

        rewards = tf.concat(rewards, axis=1)
        rewards = rewards * y_sample_mask
        rewards = rewards / (1. * roll_num)  # [batch, maxlen]
        roll_mean_loss = tf.reduce_mean(tf.concat(roll_losses, axis=1))
        real_mean_loss = tf.reduce_mean(real_loss)
        return rewards, roll_mean_loss, real_mean_loss

    def g_loss(self, gen_targets, rewards):
        """

        :param gen_targets: [batch, gen_length]
        :param given_num:
        :param rewards:  [batch, ]
        :return:
        """
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            logits = self.decode(targets=gen_targets, encoder_outputs=self.encoder_outputs,
                                 attention_bias=self.attention_bias)  # [batch, dec_len, vocab_size]
            batch_size = tf.shape(gen_targets)[0]
            probs = tf.nn.softmax(
                tf.reshape(logits, [-1, self.params.target_vocab_size]),
                axis=-1)  # probability, [batch * tgt_len, dec_vocab_size]

            log_probs = tf.reduce_sum(
                tf.one_hot(tf.reshape(gen_targets, [-1]), self.params.target_vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(probs, 1e-20, 1.0)), axis=1)

            rewards = tf.stop_gradient(rewards)

            g_loss = - tf.reduce_sum(log_probs * tf.reshape(rewards, [-1])) / tf.to_float(batch_size)
            return g_loss


class Discriminator(Transformer):
    def __init__(self, params, is_train, name_scope, mode=None):
        super(Discriminator, self).__init__(params, is_train, mode=None, scope=name_scope)
        self.name_scope = name_scope

    def get_loss(self, gen_targets, real_inputs):
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            attention_bias = model_utils.get_padding_bias(gen_targets)
            encoder_outputs = self.encode(gen_targets, attention_bias)

            logits = self.decode(real_inputs, encoder_outputs, attention_bias)

            xentropy, weights = metrics.padded_cross_entropy_loss(logits, real_inputs,
                                                                  self.params.label_smoothing,
                                                                  self.params.target_vocab_size)
            loss = tf.reduce_sum(xentropy, axis=1) / tf.reduce_sum(weights, axis=1)  # [batch, 1]
            return tf.reshape(loss, (-1, 1))


if __name__ == "__main__":
    import os
    from model import model_params
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    g_model = Generator(params, ModeKeys.TRAIN == "train", name_scope="Transformer")
    d_model = Discriminator(params, ModeKeys.TRAIN == "train", name_scope="Discriminator")
    gen_samples = g_model.inference(x_inputs, targets=None)["outputs"]
    total_rewards, roll_mean_loss, real_mean_loss = g_model.get_reward(x_inputs, y_target, gen_samples,
                                       roll_num=5, discriminator=d_model, maxlen=25)
    g_loss = g_model.g_loss(gen_samples, total_rewards)
    # print(total_rewards.shape, roll_mean_loss, real_mean_loss)
    print(g_loss)