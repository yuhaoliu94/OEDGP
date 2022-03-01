import os
import subprocess
import sys

sys.path.append(".")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from datasets import DataSet
import utils
import likelihoods
from oedgp_rff import OedgpRff
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import losses

def import_dataset(dataset, fold):

    train_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtrain__FOLD_' + fold, delimiter=' ')
    train_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytrain__FOLD_' + fold, delimiter=' ')
    test_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtest__FOLD_' + fold, delimiter=' ')
    test_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytest__FOLD_' + fold, delimiter=' ')

    data = DataSet(train_X, train_Y, dataset)
    test = DataSet(test_X, test_Y, dataset)

    return data, test

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    lines = []
    for line in iter(p.stdout.readline, b''):
        line = line.strip().decode("GB2312")
        print(">>>", line)
        lines.append(line)
    return lines

if __name__ == '__main__':
    FLAGS = utils.get_flags()

    ## Set random seed for tensorflow and numpy operations
    np.random.seed(FLAGS.seed)

    data, test = import_dataset(FLAGS.dataset, FLAGS.fold)

    ## Here we define a custom loss for dgp to show
    error_rate = losses.ZeroOneLoss(data.Dout)

    ## Likelihood
    like = likelihoods.Softmax()

    ## If use variational inference as prior
    if FLAGS.VI:
        if os.path.exists("Initialization/" + FLAGS.dataset):
            print(">>> The variational inference has been done.")
            print("")
        else:
            os.chdir("VI/")
            cmd = "python experiments/dgp_rff_classification.py  --seed=" + str(FLAGS.seed) + " --dataset=" + FLAGS.dataset + \
                  " --fold=" +str(FLAGS.fold) + " --theta_fixed=" + str(FLAGS.theta_fixed) + " --is_ard=" + str(FLAGS.is_ard) + \
                  " --optimizer=" + str(FLAGS.optimizer) + " --nl=" + str(FLAGS.nl) + " --learning_rate=" + str(FLAGS.learning_rate) + \
                  " --n_rff=" + str(FLAGS.n_rff) + " --df=" + str(FLAGS.df) + " --batch_size=" + str(FLAGS.batch_size) + \
                  " --mc_train=" + str(FLAGS.mc_train) + " --mc_test=" + str(FLAGS.mc_test) + " --n_iterations=" + str(FLAGS.n_iterations) + \
                  " --display_step=" + str(FLAGS.display_step) + " --duration=" + str(FLAGS.duration) + \
                  " --learn_Omega=" + str(FLAGS.learn_Omega) + " --local_reparam=" + str(FLAGS.local_reparam) + \
                  " --kernel_type=" + str(FLAGS.kernel_type) + " --q_Omega_fixed=" + str(FLAGS.q_Omega_fixed)
            sh(cmd)
            os.chdir("../")

    ## Main dgp object
    oedgp = OedgpRff(data, like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df,
                 FLAGS.kernel_type, FLAGS.mc_test, FLAGS.VI)

    ## Learning
    oedgp.learn(FLAGS.M, None, FLAGS.display_step, FLAGS.duration, test, FLAGS.mc_test, error_rate,
               FLAGS.less_prints)

    # odgp = OdgpRff(data, like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df,
    #              FLAGS.kernel_type, 1, FLAGS.VI)
    #
    # ## Learning
    # odgp.learn(FLAGS.M, FLAGS.n_iterations * FLAGS.batch_size, FLAGS.display_step, FLAGS.duration, test, FLAGS.mc_test, error_rate,
    #            FLAGS.less_prints)


