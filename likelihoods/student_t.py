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

from . import likelihood
import utils

class Student_t(likelihood.Likelihood):
    # def __init__(self):

    def log_cond_prob(self, output, mean, scale, nu):
        return utils.log_t_pdf(output, mean, scale, nu)

    def get_params(self):
        return None

    def predict(self, latent_val):
        return latent_val

    def get_name(self):
        return "Regression"
