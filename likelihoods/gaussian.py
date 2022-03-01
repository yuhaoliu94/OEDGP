## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
## Original code by Karl Krauth
## Changes by Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone

import tensorflow as tf

from . import likelihood
import utils


class Gaussian(likelihood.Likelihood):
    # def __init__(self, log_var=-2.0):

    def log_cond_prob(self, output, mean, scale):
        return utils.log_norm_pdf(output, mean, scale)

    def get_params(self):
        return None

    def predict(self, latent_val):
        # std = tf.exp(self.log_var / 2.0)
        return latent_val  # + std * tf.random_normal([1, tf.shape(latent_val)[1], 1])

    def get_name(self):
        return "Regression"
