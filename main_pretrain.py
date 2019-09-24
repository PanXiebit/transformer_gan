import tensorflow as tf
from model import model_params
from config import flags_obj
import os
from utils import dataset, metrics, tokenizer, compute_bleu
from six.moves import xrange  # pylint: disable=redefined-builtin
from model import transformer_9
import re
import numpy as np
from datetime import datetime

TOWER_NAME = "tower"
MOVING_AVERAGE_DECAY = 0.9999

vocab_file_source = os.path.join(flags_obj.data_dir,
                                 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.fro)
vocab_file_target = os.path.join(flags_obj.data_dir,
                                 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.to)
print(vocab_file_source, vocab_file_target)
subtokenizer_source = tokenizer.Subtokenizer.init_from_files(vocab_file_source, flags_obj.search)
subtokenizer_target = tokenizer.Subtokenizer.init_from_files(vocab_file_target, flags_obj.search)


def overwrite_params():
    PARAMS_MAP = {
        "base": model_params.TransformerBaseParams,
        "small": model_params.TransformerSmallParams,
    }

    params = PARAMS_MAP[flags_obj.param_set]
    params.data_dir = flags_obj.data_dir
    params.model_dir = flags_obj.model_dir
    params.num_parallel_calls = flags_obj.num_parallel_calls
    params.batch_size = flags_obj.batch_size or params.batch_size
    params.learning_rate = flags_obj.learning_rate or params.learning_rate
    params.max_length = flags_obj.max_length or params.max_length
    params.is_reversed = flags_obj.is_reversed
    params.extra_decode_length = flags_obj.extra_decode_length
    params.keep_checkpoint_max = flags_obj.keep_checkpoint_max
    params.save_checkpoints_secs = flags_obj.save_checkpoints_secs
    params.hvd = flags_obj.hvd
    params.repeat_dataset = -1
    params.shared_embedding_softmax_weights = flags_obj.shared_embedding_softmax_weights

    fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' + str(flags_obj.vocabulary) +
                           "." + flags_obj.fro), 'r')
    lines = fp.readlines()
    params.source_vocab_size = len(lines)
    fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' + str(flags_obj.vocabulary) +
                           "." + flags_obj.to), 'r')
    lines = fp.readlines()
    params.target_vocab_size = len(lines)

    if params.shared_embedding_softmax_weights:
        assert params.target_vocab_size == params.source_vocab_size
        params.vocab_size = params.source_vocab_size
        tf.logging.info("sharing vocab size:{}".format(params.vocab_size))
    else:
        tf.logging.info("source vocab size:{}, target vocab size:{}".format
                        (params.source_vocab_size, params.target_vocab_size))
    return params


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(global_step)
        learning_rate *= (hidden_size ** -0.5)
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        tf.identity(learning_rate, "learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)
        return learning_rate


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_loss(logits, labels, params):
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, labels, params.label_smoothing, params.target_vocab_size)
    cross_entropy_mean = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, model, input_fn, params):
    logits = model.inference(input_fn.source, input_fn.target)
    _ = get_loss(logits, input_fn.target, params)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss


def evaluation(model, input_fn, params):
    tf.logging.info("!!!Build graph for evaluation!!!")
    logits = model.inference(input_fn.source, input_fn.target)
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, input_fn.target, params.label_smoothing, params.target_vocab_size)
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    return loss, logits, input_fn.source, input_fn.target


def array_to_string(samples):
    string = ""
    for ids in samples:
        token = subtokenizer_target.subtoken_list[ids]
        string = string + token + " "
    return string


def train(params):
    with tf.Graph().as_default():
        if tf.train.latest_checkpoint(flags_obj.model_dir):
            global_step_value = int(tf.train.latest_checkpoint(flags_obj.model_dir).split("-")[-1])
            global_step = tf.Variable(
                initial_value=global_step_value,
                dtype=tf.int32,
                trainable=False)
            print("right here!", int(tf.train.latest_checkpoint(flags_obj.model_dir).split("-")[-1]))
        else:
            global_step_value = 0
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = get_learning_rate(
            params.learning_rate, params.hidden_size,
            params.learning_rate_warmup_steps,
            global_step)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params.optimizer_adam_beta1,
            beta2=params.optimizer_adam_beta2,
            epsilon=params.optimizer_adam_epsilon)

        my_dataset = dataset.Dataset(params)

        train_iterator = my_dataset.train_input_fn(params)
        valid_iterator = my_dataset.eval_input_fn(params)

        tower_grads = []
        g_model = transformer_9.Transformer(params, is_train=True, mode=None, scope="Transformer")
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i in xrange(flags_obj.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        tf.logging.info("Build graph on gpu:{}".format(i))
                        logits = g_model.inference(train_iterator.source, train_iterator.target)
                        xentropy, weights = metrics.padded_cross_entropy_loss(
                            logits, train_iterator.target, params.label_smoothing, params.target_vocab_size)
                        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = optimizer.compute_gradients(loss)
                        tf.logging.info("total trainable variables number: {}".format(len(grads)))
                        tower_grads.append(grads)
                    if i == 0 and valid_iterator:
                        valid_pred = g_model.inference(inputs=valid_iterator.source,
                                                       targets=None)["outputs"]
                        valid_tgt = valid_iterator.target
                        valid_src = valid_iterator.source

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        train_op = apply_gradient_op

        saver = tf.train.Saver(tf.trainable_variables(),
                               max_to_keep=20)

        init = tf.global_variables_initializer()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True

        with tf.Session(config=sess_config) as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())

            sess.run(train_iterator.initializer)

            ckpt = tf.train.latest_checkpoint(flags_obj.model_dir)
            tf.logging.info("ckpt {}".format(ckpt))
            if ckpt and tf.train.checkpoint_exists(ckpt):
                tf.logging.info("Reloading model parameters..from {}".format(ckpt))
                saver.restore(sess, ckpt)
            else:
                tf.logging.info("create a new model...{}".format(flags_obj.model_dir))
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(flags_obj.model_dir, sess.graph)

            count = 0
            best_bleu = 0.0
            for step in xrange(global_step_value, flags_obj.train_steps):
                _, loss_value, lr_value = sess.run([train_op, loss, learning_rate], 
                                                   feed_dict={g_model.dropout_rate: 0.1})
                if step % 200 == 0:
                    tf.logging.info("step: {}, loss = {:.4f}, lr = {:5f}".format(
                        step, loss_value, lr_value)) 
                
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step < 10000:
                    steps_between_evals = 2000
                else:
                    steps_between_evals = 1000
                if step % steps_between_evals == 0:
                    sess.run(valid_iterator.initializer)
                    tf.logging.info("------------------ Evaluation bleu -------------------------")
                    total_bleu = 0.0
                    total_size = 0
                    while True:
                        try:
                            val_pred, val_tgt, val_src = sess.run(
                                [valid_pred, valid_tgt, valid_src],
                                feed_dict={g_model.dropout_rate: 0.0})
                            val_bleu = metrics.compute_bleu(val_tgt, val_pred)
                            batch_size = val_pred.shape[0]
                            total_bleu += val_bleu * batch_size
                            total_size += batch_size
                        except tf.errors.OutOfRangeError:
                            break
                    total_bleu /= total_size
                    tf.logging.info(
                        "{}, Step: {}, Valid bleu : {:.6f}".
                            format(datetime.now(), step, total_bleu))
                    tf.logging.info(
                        "--------------------- Finish evaluation ------------------------")
                    # Save the model checkpoint periodically.
                    if step == 0:
                        total_bleu = 0.0

                    if total_bleu > best_bleu:
                        best_bleu = total_bleu
                        checkpoint_path = os.path.join(flags_obj.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        tf.logging.info("Saving model at {}".format(checkpoint_path + "-" + str(step)))
                    elif total_bleu + 0.003 > best_bleu:
                        checkpoint_path = os.path.join(flags_obj.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        tf.logging.info("Saving model at {}".format(checkpoint_path + "-" + str(step)))
                    else:
                        count += 1
                        # early stop
                        if count > 5:
                            break
            tf.logging.info("Best bleu is {}".format(best_bleu))


def main(argv=None):  # pylint: disable=unused-argument
    params = overwrite_params()
    train(params)


if __name__ == '__main__':
    #if tf.gfile.Exists(flags_obj.model_dir):
    #   tf.gfile.DeleteRecursively(flags_obj.model_dir)
    #tf.gfile.MakeDirs(flags_obj.model_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


