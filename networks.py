from random import random
from typing import Any
import copy
from functools import partial
import numpy as np
import networkx as nx
from scipy.special import expit
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn import ensemble
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.util import view_as_blocks, view_as_windows
from scipy.stats import entropy
import json
from scipy import misc

main_rng = random.PRNGKey(42)

#lil-net
class LateralInhibitoryLayer:
    """
    A network that implements the original lateral inhibition algorithm to learn a factorized representation
    of some input by updating both weights using local plasticity rules.
    
    """

    @staticmethod
    def get_matrices():
        """
        Returns the list of matrices that are used in the network
        """
        return ['w', 'm', 'p']

    @staticmethod
    def default_hparams(input_shape):
        hparams = dict()
        hparams['name'] = 'lil-net'
        hparams['input_shape'] = input_shape
        hparams['num_units_y'] = 500
        hparams['num_blocks'] = 1
        hparams['step'] = 0.01
        hparams['num_steps'] = 1000
        hparams['multisteps'] = 1
        hparams['activation'] = 'relu'
        hparams['threshold_dynamics'] = 0.0 # should be 0 for pure relu
        hparams['gradient'] = 'voltage' #voltage, rate
        hparams['spont_activity'] = 0.0
        hparams['bias'] = 'zero_vector'
        hparams['threshold_learning'] = 0.0
        hparams['bcm_exprss'] = 'y_mean'
        hparams['noisy_lr_std'] = 0.001
        hparams['dynamic_lr_exprss'] = 'pehlevan_dynamic_lr'
        hparams['dynamic_lr_init'] = 1000
        hparams['foldiak_constant'] = 1.0
        hparams['zero_sequence'] = 'zero'
        hparams['zero_epoch'] = 'none'
        hparams['modulator'] = 'self.y'
        hparams['memory_size'] = 10
        hparams['batch_size'] = 32
        hparams['w_init'] = 'random'
        hparams['w_seed'] = np.random.randint(0, 100000)
        hparams['w_update'] = 'bcm_abs'
        hparams['w_update_sleep'] = 'bcm_abs'
        hparams['w_lr'] = 0.01
        hparams['w_noisy'] = False
        hparams['w_dynamic'] = False
        hparams['w_rectify'] = True
        hparams['w_kroned'] = 'no'
        hparams['w_batched'] = False
        hparams['w_gate'] = False
        hparams['m_init'] = 'random'
        hparams['m_seed'] = np.random.randint(0, 100000)
        hparams['m_update'] = 'hebbian'
        hparams['m_update_sleep'] = 'hebbian'
        hparams['m_lr'] = 0.01
        hparams['m_noisy'] = False
        hparams['m_dynamic'] = False
        hparams['m_rectify'] = True
        hparams['m_zero_dig'] = False
        hparams['m_kroned'] = 'no'
        hparams['m_batched'] = False
        hparams['m_gate'] = False
        hparams['p_init'] = 'zero'
        hparams['p_seed'] = np.random.randint(0, 100000)
        hparams['p_update'] = 'hebbian'
        hparams['p_update_sleep'] = 'hebbian'
        hparams['p_lr'] = 0.01
        hparams['p_noisy'] = False
        hparams['p_dynamic'] = False
        hparams['p_rectify'] = True
        hparams['p_kroned'] = 'no'
        hparams['p_zero_dig'] = False
        hparams['p_batched'] = False
        hparams['p_gate'] = False
        hparams['use_jax'] = 'off'
        hparams['sleep_sparsity'] = 0.1
        hparams['target_activity'] = 5.0
        hparams['target_activity_lr'] = 0.001
        hparams['center_input'] = False
        hparams['gating_threshold'] = '0.0'
        hparams['normalize_input'] = False
        hparams['normalize_input_historically'] = False
        hparams['nih_mean'] = 1.0
        hparams['nih_variance'] = 1.0
        hparams['normalize_w_weights'] = False
        hparams['normalize_p_weights'] = False
        hparams['normalize_m_weights'] = False
        return hparams

    def __init__(self, hparams):
        self.name = hparams['name']

        self.y = np.zeros(hparams['num_units_y'])
        self.previous_y = np.zeros(hparams['num_units_y'])

        self.iteration = 1
        
        self.num_units_y = hparams['num_units_y']
        self.num_assemblies = hparams['num_blocks']
        assert (self.num_units_y % self.num_assemblies) == 0
        self.inp_shape = hparams['input_shape']
        self.inp_size = np.prod(hparams['input_shape'])

        self.y_activity_track = np.zeros((hparams['num_steps'], self.num_units_y))

        self.activation = hparams['activation']
        self.threshold = hparams['threshold_dynamics']

        self.zero_sequence = hparams['zero_sequence']
        self.zero_epoch = hparams['zero_epoch']

        self.bias_expression = hparams['bias']

        self.track_input = np.zeros(self.inp_size)
        self.track_y = np.zeros(self.num_units_y)
        self.track_y_squared = np.zeros(self.num_units_y)

        self.y_exp_mean = np.zeros(self.num_units_y)

        self.memory_size = hparams['memory_size']

        self.normalize_input = hparams['normalize_input']
        self.normalize_input_historically = hparams['normalize_input_historically']
        self.nih_decay = 0.001
        self.nih_mean = hparams['nih_mean']
        self.nih_variance = hparams['nih_variance']
        self.running_mean = np.zeros(self.inp_size)
        self.running_var = np.zeros(self.inp_size)

        self.wx_raw_window = np.zeros((hparams['num_units_y'], hparams['memory_size']))
        self.py_raw_window = np.zeros((hparams['num_units_y'], hparams['memory_size']))
        self.y_window = np.zeros((hparams['num_units_y'], hparams['memory_size']))
        
        self.y_gradient_track = list()
        self.y_sparsity_track = list()
        self.y_entropy_track = list()

        self.init_w = hparams['w_init']
        self.init_p = hparams['p_init']
        self.init_m = hparams['m_init']
        


        np.random.seed(hparams['w_seed'])
        if self.init_w == 'random':
            self.w = np.random.rand(self.num_units_y, self.inp_size)
        elif self.init_w == 'random_normal':
            self.w = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y, self.inp_size))
        elif self.init_w == 'zero':
            self.w = np.zeros((self.num_units_y, self.inp_size))
        else:
            raise Exception('No valid initial value for W provided')


        np.random.seed(hparams['p_seed'])
        if self.init_p == 'random':
            self.p = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_p == 'zero':
            self.p = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception('No valid initial value for P provided')
        

        np.random.seed(hparams['m_seed'])
        if self.init_m == 'random':
            self.m = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_m == 'random_normal':
            self.m = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y, self.num_units_y))
        elif self.init_m == 'posdef':
            B = np.random.randn(self.num_units_y, self.num_units_y)
            self.m = B.T @ B 
        elif self.init_m == 'wwt':
            self.m = self.w @ self.w.T
        elif self.init_m == 'identity':
            self.m = np.identity(self.num_units_y)
        elif self.init_m == 'zero':
            self.m = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception('No valid initial value for M provided')
        
        
        #DYNAMIC LEARNING RATES
        self.threshold_learning = hparams['threshold_learning']
        self.dyn_lr_init = hparams['dynamic_lr_init']
        self.pehvelan_lr = np.zeros(self.num_units_y)
        self.pehvelan_lr.fill(self.dyn_lr_init)
        self.dynamic_lr_increment_exprss = hparams['dynamic_lr_exprss']
        self.modulate_stable_y_exprss = hparams['modulator']

        self.bias_perceptron = np.zeros(self.num_units_y)
        self.bias_target = np.ones(self.num_units_y) * hparams['target_activity']
        self.target_activity_lr = hparams['target_activity_lr']
        self.homeostatic_bias = np.zeros(self.num_units_y)
        
        #BCM
        self.bcm_trsh_exp = hparams['bcm_exprss']

        #
        self.foldiak_constant = hparams['foldiak_constant']

        self.num_steps = hparams['num_steps']
        self.multisteps = hparams['multisteps']
        self.step_size_y = hparams['step']
        self.gradient = hparams['gradient']

        self.spont_act = hparams['spont_activity']

        self.lr_w = hparams['w_lr']
        self.lr_p = hparams['p_lr']
        self.lr_m = hparams['m_lr']

        self.rule_w = hparams['w_update']
        self.rule_p = hparams['p_update']
        self.rule_m = hparams['m_update']

        self.dynamic_w = hparams['w_dynamic']
        self.dynamic_p = hparams['p_dynamic']
        self.dynamic_m = hparams['m_dynamic']

        self.noisy_w = hparams['w_noisy']
        self.noisy_p = hparams['p_noisy']
        self.noisy_m = hparams['m_noisy']

        self.noise_std_w = hparams['noisy_lr_std']
        self.noise_std_p = hparams['noisy_lr_std']
        self.noise_std_m = hparams['noisy_lr_std']

        self.rectify_w = hparams['w_rectify']
        self.rectify_p = hparams['p_rectify']
        self.rectify_m = hparams['m_rectify']

        self.kroned_w = hparams['w_kroned']
        self.kroned_p = hparams['p_kroned']
        self.kroned_m = hparams['m_kroned']

        self.batched_w = hparams['w_batched']
        self.batched_p = hparams['p_batched']
        self.batched_m = hparams['m_batched']

        self.gate_w = hparams['w_gate']
        self.gate_m = hparams['m_gate']
        self.gate_p = hparams['p_gate']
        self.gate_threshold = hparams['gating_threshold']
        self.center_input = hparams['center_input']

        self.normalize_w_weights = hparams['normalize_w_weights']
        self.normalize_p_weights = hparams['normalize_p_weights']
        self.normalize_m_weights = hparams['normalize_m_weights']

        self.accumulated_gradient_w = np.zeros((self.num_units_y, self.inp_size))
        self.accumulated_gradient_p = np.zeros((self.num_units_y, self.num_units_y))
        self.accumulated_gradient_m = np.zeros((self.num_units_y, self.num_units_y))

        self.zero_dig_p = hparams['p_zero_dig']
        self.zero_dig_m = hparams['m_zero_dig']

        self.sleep_sparsity = hparams['sleep_sparsity']

        self.batch_size = hparams['batch_size']

        self.rng = jax.random.PRNGKey(np.random.randint(0, 100000))
        
        if "use_jax" in hparams:
            self.use_jax = hparams['use_jax']
        else:
            self.use_jax = "off"
        
        self.matrices = ['w', 'p', 'm']
        self.normalize_weights(self.matrices)
    
    def set_learned_params(self, saved_w, saved_m, saved_p, saved_track_input, saved_track_y, saved_track_y_squared, saved_iterations):
        self.iteration = saved_iterations
        self.track_input = saved_track_input
        self.track_y = saved_track_y
        self.track_y_squared = saved_track_y_squared
        
        self.w = saved_w
        self.m = saved_m
        self.p = saved_p

    def gradient_step(self, add_feedback_y, bias_correction_y):
        gradient_y = add_feedback_y - (self.m @ self.y) + (np.random.normal(loc=0.0, scale=self.spont_act, size=(self.num_units_y))) - bias_correction_y
        self.y += self.step_size_y * gradient_y

        if self.activation == 'relu':
            self.y = np.maximum(self.y, 0.0)
        elif self.activation == 'sigmoid':
            self.y = 1 / (1 + np.exp(-self.y))
        elif self.activation == 'linear':
            pass
        elif self.activation == 'exp':
            self.y = np.exp(self.y)
        elif self.activation == 'tanh':
            self.y = np.tanh(self.y)
        else:
            raise Exception('No valid activation function provided')
        
        return np.sum(np.abs(gradient_y))
    
    def gradient_step_rate(self, add_feedback_y, bias_correction_y):
        gradient_y = add_feedback_y - (self.m @ self.y) + (np.random.normal(loc=0.0, scale=self.spont_act, size=(self.num_units_y))) - self.threshold

        if self.activation == 'relu':
            gradient_y = np.maximum(gradient_y, 0.0)
        elif self.activation == 'sigmoid':
            gradient_y = 1 / (1 + np.exp(-gradient_y))
        elif self.activation == 'linear':
            pass
        elif self.activation == 'exp':
            gradient_y = np.exp(gradient_y)
        elif self.activation == 'tanh':
            gradient_y = np.tanh(gradient_y)
        else:
            raise Exception('No valid activation function provided')

        gradient_y = gradient_y - bias_correction_y
        self.y += self.step_size_y * gradient_y

        return np.sum(np.abs(gradient_y))

    def projection(self):
        if self.kroned_w == 'yin':
            original_squared_image_size = int(np.sqrt(self.inp_size))
            num_blocks_per_side = np.sqrt(self.num_assemblies)
            assert num_blocks_per_side.is_integer(), 'Number of blocks per side is not an integer'
            num_blocks_per_side = int(num_blocks_per_side)
            block_size = int(original_squared_image_size/num_blocks_per_side)
            frame_block_single_vector = self.w.transpose() @ self.y
            original_blocks = np.reshape(frame_block_single_vector, (num_blocks_per_side,num_blocks_per_side,block_size,block_size))
            list_of_rows = []
            for i in range(0, num_blocks_per_side):
                row = original_blocks[i][0]
                for j in range(1, num_blocks_per_side):
                    row = np.concatenate((row, original_blocks[i][j]), axis=1)
                list_of_rows.append(row)
            np.concatenate(list_of_rows, axis=0)
            image_rec = np.concatenate(list_of_rows, axis=0)
            return np.reshape(image_rec, self.inp_shape)
        elif self.kroned_w == 'delayed':
            projection_flat = self.w.transpose() @ self.y
            return np.reshape(projection_flat[-self.frame_size:], self.inp_shape)
        else:
            if self.center_input:
                input_mean = np.copy(self.track_input)/self.iteration
                projection_flat = (self.w.transpose() @ self.y) + input_mean
            else:
                projection_flat = self.w.transpose() @ self.y
            return np.reshape(projection_flat, self.inp_shape)
    
    def update_activations(self, current_input, train):
        if np.min(current_input) < 0:
            print('CAREFUL INPUT IS NEGATIVE! RUNNING MEAN WILL BE WRONG!')
            pass
        
        if self.kroned_w == 'yin':
            num_blocks_per_side = np.sqrt(self.num_assemblies)
            block_size = int(self.inp_shape[0]/num_blocks_per_side)
            original_image_view_as_blocks = view_as_blocks(current_input, (block_size, block_size))
            input_vector = np.array(original_image_view_as_blocks, dtype=np.float64)
            self.current_input = np.reshape(input_vector, (self.num_assemblies*block_size*block_size*self.inp_shape[2]))
        elif self.kroned_w == 'delayed':
            self.current_input[-self.frame_size:] = np.copy(current_input.flatten())
        else:
            self.current_input = current_input.flatten()


        zero_vector = np.zeros(self.num_units_y)
        input_mean = np.copy(self.track_input)/self.iteration
        y_mean = np.copy(self.track_y)/self.iteration
        y_squared_mean = np.copy(self.track_y_squared)/self.iteration

        if self.normalize_input_historically:
            diff = self.current_input - self.running_mean
            self.running_mean += self.nih_decay * diff
            self.running_var += self.nih_decay * ((diff * (self.current_input - self.running_mean)) - self.running_var)
            std = np.sqrt(np.maximum(self.running_var, 1e-8))
            standardized = (self.current_input - self.running_mean) / std
            # 3. Scale to target variance of 0.5 and shift to target mean of 1.0
            target_variance = self.nih_variance
            target_mean = self.nih_mean
            final_output = (standardized * np.sqrt(target_variance)) + target_mean
            self.current_input = np.maximum(final_output, 0.0)
        else:
            self.current_input = np.copy(self.current_input)
        
        if self.normalize_input:
            if (self.rule_p == 'none') and (self.init_p == 'zero'):
                norm_input = np.linalg.norm(self.current_input)
                self.current_input = (self.current_input / norm_input) * np.pi
            elif (self.rule_p == 'perceptron-single'):
                norm_input = np.linalg.norm(self.current_input)
                norm_prev = np.linalg.norm(self.previous_y)
                self.current_input = (self.current_input / norm_input) * np.pi
                self.previous_y = (self.previous_y / norm_prev) * np.pi
            else:
                norm_sq_input = np.sum(np.square(self.current_input))
                norm_sq_prev = np.sum(np.square(self.previous_y))
                joint_norm = np.sqrt(norm_sq_input + norm_sq_prev) + 1e-9
                self.current_input = (self.current_input / joint_norm) * np.pi
                self.previous_y = (self.previous_y / joint_norm) * np.pi
        else:
            self.current_input = current_input

        self.y_gradient_track = list()
        self.y_activity_track = list()
        self.y_sparsity_track = list()
        self.y_entropy_track = list()

        

        if self.center_input:
            self.current_input_y = (self.w @ (self.current_input-input_mean)) + (self.p @ (self.previous_y - y_mean)) + self.bias_perceptron
        else:
            self.current_input_y = (self.w @ self.current_input) + (self.p @ self.previous_y) + self.bias_perceptron
        if self.gradient == 'voltage':
            for i in range(0, self.num_steps*self.multisteps):
                gmag_y = self.gradient_step(self.current_input_y, eval(self.bias_expression))
                self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == 'voltage-jax':
            jax_current_input = jnp.array(np.copy(self.current_input), dtype=jnp.float64)
            jax_current_y_feedback = jnp.array(np.copy(self.previous_y), dtype=jnp.float64)
            jax_current_y = jnp.array(np.copy(self.y), dtype=jnp.float64)
            jax_w = jnp.array(np.copy(self.w), dtype=jnp.float64)
            jax_m = jnp.array(np.copy(self.m), dtype=jnp.float64)
            jax_p = jnp.array(np.copy(self.p), dtype=jnp.float64)
            jax_step = jnp.array(self.step_size_y, dtype=jnp.float64)
            jax_thrsh = jnp.array(self.threshold, dtype=jnp.float64)
            jax_num_steps = int(self.num_steps)
            jax_bias_y = jnp.array(eval(self.bias_expression), dtype=jnp.float64)
            big_gy = jnp.zeros((self.num_steps*self.multisteps), dtype=jnp.float64)
            big_ey = jnp.zeros((self.num_steps*self.multisteps), dtype=jnp.float64)
            self.rng, subkey = jax.random.split(self.rng)
            for mult_stp_idx in range(0, self.multisteps):
                y_jax, y_gradient_track_jax, y_entropy_track_jax = lil_dynamics_relu_jit(jax_current_input, jax_current_y_feedback, jax_current_y, jax_w, jax_m, jax_p, jax_step, jax_thrsh, jax_num_steps, jax_bias_y)
                y_jax.block_until_ready()
                jax_current_y = jnp.array(y_jax, dtype=jnp.float64)
                start_idx = mult_stp_idx*self.num_steps
                end_idx = (mult_stp_idx+1)*self.num_steps
                big_gy = big_gy.at[start_idx:end_idx].set(y_gradient_track_jax)
                big_ey = big_ey.at[start_idx:end_idx].set(y_entropy_track_jax)
            self.y = np.array(y_jax)
            self.y_gradient_track = np.array(big_gy)
            self.y_entropy_track = np.array(big_ey)
        elif self.gradient == 'voltage-predictive':
            for i in range(0, self.num_steps*self.multisteps):
                gmag_y = self.gradient_step(2*self.current_input_y, eval(self.bias_expression))
                self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == 'voltage-centered':
            self.current_input_y = (self.w @ (self.current_input-input_mean)) + (self.p @ (self.previous_y - y_mean)) + self.bias_perceptron
            for i in range(0, self.num_steps*self.multisteps):
                gmag_y = self.gradient_step(self.current_input_y, eval(self.bias_expression))
                self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == 'rate':
            for i in range(0, self.num_steps*self.multisteps):
                gmag_y = self.gradient_step_rate(self.current_input_y, eval(self.bias_expression))
                self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == 'predictive':
            for i in range(0, self.num_steps*self.multisteps):
                gmag_y = self.gradient_step(2*self.current_input_y, eval(self.bias_expression))
                self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == 'direct':
            self.y = np.linalg.solve(self.m, self.current_input_y)
            self.y_sparsity_track = np.zeros(self.num_steps*self.multisteps)
            self.y_activity_track = np.zeros((self.num_steps*self.multisteps, self.num_units_y))
            self.y_entropy_track = np.zeros(self.num_steps*self.multisteps)
            self.y_gradient_track = np.zeros(self.num_steps*self.multisteps)
        elif self.gradient == 'direct_prince':
            self.y = np.linalg.solve((self.m)+(0.5*np.eye(self.num_units_y)), self.current_input_y)
            self.y_sparsity_track = np.zeros(self.num_steps*self.multisteps)
            self.y_activity_track = np.zeros((self.num_steps*self.multisteps, self.num_units_y))
            self.y_entropy_track = np.zeros(self.num_steps*self.multisteps)
            self.y_gradient_track = np.zeros(self.num_steps*self.multisteps)
        elif self.gradient == 'direct_predictive':
            self.y = np.linalg.solve((self.m+np.eye(self.num_units_y)+np.eye(self.num_units_y)), 2*self.current_input_y)
            self.y_sparsity_track = np.zeros(self.num_steps*self.multisteps)
            self.y_activity_track = np.zeros((self.num_steps*self.multisteps, self.num_units_y))
            self.y_entropy_track = np.zeros(self.num_steps*self.multisteps)
            self.y_gradient_track = np.zeros(self.num_steps*self.multisteps)
        else:
            raise Exception('No valid gradient function provided')

        self.y = eval(self.modulate_stable_y_exprss)

        return np.copy(self.y)
    
    def update_activations_blank(self):
        self.update_activations(np.zeros(self.inp_shape), False)

    def update_sleep(self):
        zero_vector = np.zeros(self.num_units_y)

        if self.iteration > 0:
            y_mean = np.copy(self.track_y)/self.iteration
            y_squared_mean = np.copy(self.track_y_squared)/self.iteration
        else:
            y_mean = zero_vector
            y_squared_mean = zero_vector

        self.y_gradient_track = list()
        self.y_activity_track = list()
        self.y_sparsity_track = list()


        random_population_input = np.random.choice([0, 1], size=self.num_units_y, p=[1-self.sleep_sparsity, self.sleep_sparsity])

        
        for i in range(0, self.num_steps*self.multisteps):
            gmag_y = self.gradient_step(random_population_input, eval(self.bias_expression))
            self.y_gradient_track.append(np.sum(np.abs(gmag_y)))
            self.y_sparsity_track.append(1-(np.sum(self.y>0.0)/self.num_units_y))
            self.y_activity_track.append(np.copy(self.y))
        
        self.current_input = self.w.transpose() @ self.y

        return np.copy(self.y)

    #called at beginning of training/testing
    def reset_network(self):
        self.y = np.zeros(self.num_units_y)
        self.previous_y = np.zeros(self.num_units_y)
        self.y_window = np.zeros((self.num_units_y, self.memory_size))
    
    #called at the beginning of each sequence
    def reset_activations(self):
        if self.zero_sequence == 'zero':
            self.y.fill(0.0)
        elif self.zero_sequence == 'randu':
            self.y = np.random.uniform(low=0, high=1, size=(self.num_units_y))
        elif self.zero_sequence == 'randn':
            self.y = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y))
        else:
            pass
        self.previous_y = np.zeros(self.num_units_y)
        self.y_window = np.zeros((self.num_units_y, self.memory_size))
     
    def new_epoch(self):
        self.y_window = np.roll(self.y_window, 1, axis=1)
        self.y_window[:, 0] = np.copy(self.y)
        self.previous_y = np.sum(self.y_window, axis=1)/self.memory_size
        if self.zero_epoch == 'zero':
            self.y.fill(0.0)
        elif self.zero_epoch == 'randu':
            self.y = np.random.uniform(low=0, high=1, size=(self.num_units_y))
        elif self.zero_epoch == 'randn':
            self.y = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y))
        elif self.zero_epoch == 'none':
            pass
        else:
            pass
    
    def last_epoch(self, active_matrices):
        pass

    def update_counts(self):
        self.iteration+=1

        #Compute counts and means
        self.track_input += self.current_input
        self.track_y += self.y
        self.track_y_squared += self.y**2

        # compute new windows
        self.wx_raw_window = np.roll(self.wx_raw_window, 1, axis=1)
        self.wx_raw_window[:, 0] = (self.w @ self.current_input)
        self.py_raw_window = np.roll(self.py_raw_window, 1, axis=1)
        self.py_raw_window[:, 0] = (self.p @ self.previous_y)

    def update_parameters(self, active_matrices):

        #compute means
        input_mean = np.copy(self.track_input)/self.iteration
        y_mean = np.copy(self.track_y)/self.iteration
        y_raw_mean = np.mean(self.py_raw_window, axis=1) + np.mean(self.wx_raw_window, axis=1)
        y_squared_mean = np.copy(self.track_y_squared)/self.iteration
        self.y_exp_mean = self.y_exp_mean + self.lr_m*(self.y - self.y_exp_mean)

        if self.center_input:
            self.current_input = self.current_input - input_mean
            self.previous_y = self.previous_y - y_mean

        current_network_input = (self.w @ self.current_input) + (self.p @ self.previous_y) + self.bias_perceptron

        current_y = np.copy(self.y)

        #Compute BCM
        bcm_threshold = eval(self.bcm_trsh_exp)

        #Compute dynamic learning rates
        pehlevan_dynamic_lr = np.copy((self.track_y_squared*0.01)+self.dyn_lr_init)
        self.dynamic_learning_rate = eval(self.dynamic_lr_increment_exprss)
        
        gradient_list = list()

        self.homeostatic_bias = self.homeostatic_bias + self.target_activity_lr*(self.bias_target - self.y)

        if isinstance(self.gate_threshold, float) or isinstance(self.gate_threshold, int):
            spikes_y = self.y > self.gate_threshold
        else:
            spikes_y = self.y > eval(self.gate_threshold)

        #Update W (x -> y)
        if "w" in active_matrices:
            #correlation rule (classic hebbian rule)
            if self.rule_w == 'hebbian': #delta wij = (y_i * x_j) - w_ij
                gradient_w = np.outer(self.y, self.current_input) - self.w
            #chklovskii rule (classic hebbian rule with mean normalization)
            #foldiak rule (classic hebbian rule with weighted forgetting factor)
            elif self.rule_w == 'foldiak': #delta wij = (y_i * x_j) - (w_ij * y_i)
                gradient_w = np.outer(self.y, self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            #input reconstruction rule (classic hebbian rule with heavily weighted forgetting factor)
            elif (self.rule_w == 'oja') or (self.rule_w == 'pehlevan'): #delta wij = (y_i * x_j) - (w_ij * y_i^2)
                gradient_w = np.outer(self.y, self.current_input) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'chklovskii':  #delta wij = (y_i - y_mean) * (x_j - x_mean) - w_ij
                gradient_w = np.outer(self.y - y_mean, self.current_input - input_mean) - self.w
            #approximate stable state rule (objective derived rule to lead feedforward weigths to approximate dynamics)
            elif (self.rule_w == 'reconstruction'):
                gradient_w = np.outer(self.y, (self.current_input - (self.w.T @ self.y)))
            elif self.rule_w == 'balancer':
                gradient_w = np.outer(((self.w @ self.current_input) - self.y), self.current_input)
            elif (self.rule_w == 'perceptron') or (self.rule_w == 'perceptron-single') or (self.rule_w == 'perceptron-refractory'): #delta wij = 
                gradient_w = np.outer((self.y - (current_network_input)), self.current_input)
            elif self.rule_w == 'perceptron_double': #delta wij = 
                gradient_w = np.outer(((2*self.y) - (current_network_input)), self.current_input)
            elif self.rule_w == 'perceptron_mean': #delta wij =
                gradient_w = np.outer((self.y - (current_network_input) - y_mean), self.current_input)
            elif self.rule_w == 'perceptron_bias': #delta wij = 
                self.bias_perceptron = self.bias_perceptron + self.lr_w*(self.y - current_network_input)
                gradient_w = np.outer((self.y - (current_network_input)), self.current_input)
            elif self.rule_w == 'perceptron_chklovskii': #delta wij = 
                gradient_w = np.outer((self.y - (current_network_input)), self.current_input-input_mean)
            elif self.rule_w == 'perceptron_norm': #delta wij = 
                gradient_w = np.outer((self.y - ((self.w @ (self.current_input-input_mean)))), self.current_input-input_mean)
            elif self.rule_w == 'perceptron_modulated': #delta wij = 
                gradient_w = np.outer(np.multiply(self.y, (self.y - (current_network_input))), self.current_input)
            elif self.rule_w == 'predictive': #delta wij = 
                gradient_w = np.outer(((2*self.y) - (current_network_input) - y_mean), self.current_input)
            elif self.rule_w == 'predictive_bias': #delta wij = 
                self.bias_perceptron = self.bias_perceptron + self.lr_w*((2*self.y) - current_network_input)
                gradient_w = np.outer(((2*self.y) - (current_network_input)), self.current_input)
            elif self.rule_w == 'perceptron_leaky':
                gradient_w = np.outer((self.y - (self.w @ self.current_input)), self.current_input) - self.w
            elif self.rule_w == 'perceptron_leaky_foldiak':
                gradient_w = np.outer((self.y - (self.w @ self.current_input)), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'perceptron_leaky_oja':
                gradient_w = np.outer((self.y - (self.w @ self.current_input)), self.current_input) - np.multiply(np.repeat(np.array([np.multiply((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)]), self.inp_size, axis=0).T,self.w)
            #bcm absolute with arbitrary threshold
            elif self.rule_w == 'pure_bcm_abs':
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input)
            elif self.rule_w == 'pure_bcm_abs_constant':
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input) - self.foldiak_constant**2
            elif self.rule_w == 'bcm_abs': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input) - self.w
            elif self.rule_w == 'bcm_abs_chklovskii': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input-input_mean) - self.w
            elif self.rule_w == 'bcm_abs_chklovskii_abs': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), (np.abs(self.current_input - (input_mean/2)) - (input_mean/2))) - self.w
            elif self.rule_w == 'bcm_abs_foldiak': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'bcm_abs_oja': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'bcm_abs_krotov': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.current_input) - np.multiply(np.repeat(np.array([np.multiply((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)]), self.inp_size, axis=0).T,self.w)
            
            #bcm linear with arbitrary threshold
            elif self.rule_w == 'pure_bcm_linear':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input)
            elif self.rule_w == 'pure_bcm_linear_constant':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input) - self.foldiak_constant**2
            elif self.rule_w == 'bcm_linear':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input) - self.w         
            elif self.rule_w == 'bcm_linear_foldiak':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'bcm_linear_oja':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'bcm_linear_krotov':
                gradient_w = np.outer((self.y - bcm_threshold), self.current_input) - np.multiply(np.repeat(np.array([np.multiply((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)]), self.inp_size, axis=0).T,self.w)
            
            #bcm abs with norm threshold
            elif self.rule_w == 'pure_bcm_abs_norm':
                bcm_threshold_norm = np.linalg.norm(self.w, axis=1)
                gradient_w = np.outer((np.abs(self.y - bcm_threshold_norm) - bcm_threshold_norm), self.current_input)
            elif self.rule_w == 'bcm_abs_norm':
                bcm_threshold_norm = np.linalg.norm(self.w, axis=1)
                gradient_w = np.outer((np.abs(self.y - bcm_threshold_norm) - bcm_threshold_norm), self.current_input) - self.w
            elif self.rule_w == 'bcm_abs_norm_foldiak':
                bcm_threshold_norm = np.linalg.norm(self.w, axis=1)
                gradient_w = np.outer((np.abs(self.y - bcm_threshold_norm) - bcm_threshold_norm), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            elif self.rule_w == 'bcm_abs_norm_oja':
                bcm_threshold_norm = np.linalg.norm(self.w, axis=1)
                gradient_w = np.outer((np.abs(self.y - bcm_threshold_norm) - bcm_threshold_norm), self.current_input) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.inp_size, axis=0).T,self.w)
            
            
            #bcm rule square
            elif self.rule_w == 'bcm_sqr':
                gradient_w = np.outer(np.multiply(self.y, self.y) - np.multiply(self.y, bcm_threshold), self.current_input) - self.w
            #bcm rule threshold 
            elif self.rule_w == 'bcm_thrs':
                gradient_w = np.outer(np.abs(self.y - bcm_threshold), self.current_input) - self.w
            else:
                raise Exception("Rule W not implemented")
            
            gradient_list.append(gradient_w)
            
            if self.gate_w:
                gradient_w_T_modulated = np.multiply(gradient_w.T, spikes_y)
                gradient_w = gradient_w_T_modulated.T

            if self.dynamic_w or (self.rule_w == 'pehlevan'):
                self.w += (1/self.dynamic_learning_rate.reshape((self.num_units_y,1)))*gradient_w
            elif self.batched_w:
                if (self.iteration % self.batch_size) != 0:
                    self.accumulated_gradient_w += gradient_w
                else:
                    self.accumulated_gradient_w += gradient_w
                    self.w += self.lr_w*(self.accumulated_gradient_w/self.batch_size)
                    self.accumulated_gradient_w = np.zeros((self.num_units_y, self.inp_size))
            elif self.rule_w == 'perceptron-refractory':
                if (self.iteration % 50) == 0:
                    self.w += self.lr_w*gradient_w
            else:
                self.w += self.lr_w*gradient_w
            
            if self.noisy_w:
                self.w += norm.ppf(np.random.rand(self.num_units_y, self.inp_size))*np.sqrt(self.lr_w)*self.noise_std_w

            if self.rectify_w:
                self.w[self.w < self.threshold_learning] = 0.0
            
            if self.kroned_w == 'yin':
                self.w = kron_matrix_yin(self.w, self.num_assemblies)
            elif self.kroned_w == 'delayed':
                self.w = kron_matrix_yin(self.w, self.num_assemblies)
            elif self.kroned_w == 'yang':
                self.w = kron_matrix_yang(self.w, self.num_assemblies)
            else:
                pass
        
        
        #Update P (prev_y -> y)  
        if "p" in active_matrices:
            #correlation rule (classic hebbian rule)
            if self.rule_p == 'hebbian': #delta pij = (y_i * x_j) - p_ij
                gradient_p = np.outer(self.y, self.previous_y) - self.p
            #chklovskii rule (classic hebbian rule with mean normalization)
            elif self.rule_p == 'chklovskii':  #delta pij = (y_i - y_mean) * (previous_y_i - y_mean) - p_ij
                gradient_p = np.outer(self.y - y_mean, self.previous_y - y_mean) - self.p
            #foldiak rule (classic hebbian rule with weighted forgetting factor)
            elif self.rule_p == 'foldiak': #delta pij = (y_i * previous_y_j) - (p_ij * y_i)
                gradient_p = np.outer(self.y, self.previous_y) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.num_units_y, axis=0).T,self.p)
            #input reconstruction rule (classic hebbian rule with heavily weighted forgetting factor)
            elif (self.rule_p == 'oja') or (self.rule_p == 'pehlevan'): #delta pij = (y_i * previous_y_j) - (w_ij * y_i^2)
                gradient_p = np.outer(self.y, self.previous_y) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.num_units_y, axis=0).T,self.p)
            elif (self.rule_p == 'reconstruction'):
                gradient_p = np.outer(self.y, (self.previous_y - (self.p.T @ self.y)))
            #approximate stable state rule (objective derived rule to lead feedforward weigths to approximate dynamics)
            elif (self.rule_p == 'perceptron') or (self.rule_p == 'perceptron-single'): #delta pij = (y_i * previous_y_j) - (w_ij * previous_y_j^2)
                #raw_y = (self.w @ self.current_input) + (self.p @ self.previous_y)
                gradient_p = np.outer((self.y - (current_network_input)), self.previous_y)
            elif self.rule_p == 'perceptron_modulated': #delta pij = (y_i * previous_y_j) - (w_ij * previous_y_j^2)
                gradient_p = np.outer(np.multiply(self.y, (self.y - (current_network_input))), self.previous_y)
            elif self.rule_p == 'predictive': #delta pij = (2*y_i * previous_y_j) - (w_ij * previous_y_j^2)
                gradient_p = np.outer(((2*self.y) - (current_network_input)), self.previous_y)
            #bcm rule absolute value
            elif self.rule_p == 'bcm_abs':
                gradient_p = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.previous_y) - self.p
            elif self.rule_p == 'bcm_abs_foldiak': #delta pij = (|y_i-y_i_mean| - y_i) * x_j) - p_ij
                gradient_p = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.previous_y) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.num_units_y, axis=0).T,self.p)
            elif self.rule_p == 'bcm_abs_oja': #delta pij = (|y_i-y_i_mean| - y_i) * x_j) - p_ij
                gradient_p = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.previous_y) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.num_units_y, axis=0).T,self.p)
            elif self.rule_p == 'bcm_abs_krotov': #delta pij = (|y_i-y_i_mean| - y_i) * x_j) - p_ij
                gradient_p = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.previous_y) - np.multiply(np.repeat(np.array([np.multiply((np.abs(self.y - (bcm_threshold/2)) - bcm_threshold), self.y)]), self.num_units_y, axis=0).T,self.p)
            #bcm rule absolute value without forgetting factor
            elif self.rule_p == 'pure_bcm_abs':
                gradient_p = np.outer((np.abs(self.y - (bcm_threshold/2)) - (bcm_threshold/2)), self.previous_y)
            
            elif self.rule_p == 'pure_bcm_linear':
                gradient_p = np.outer((self.y - bcm_threshold), self.previous_y)
            elif self.rule_p == 'bcm_linear':
                gradient_p = np.outer((self.y - bcm_threshold), self.previous_y) - self.p         
            elif self.rule_p == 'bcm_linear_foldiak':
                gradient_p = np.outer((self.y - bcm_threshold), self.previous_y) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.num_units_y, axis=0).T,self.p)
            elif self.rule_p == 'bcm_linear_oja':
                gradient_p = np.outer((self.y - bcm_threshold), self.previous_y) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.num_units_y, axis=0).T,self.p)
            elif self.rule_p == 'bcm_linear_krotov':
                gradient_p = np.outer((self.y - bcm_threshold), self.previous_y) - np.multiply(np.repeat(np.array([np.multiply((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)]), self.num_units_y, axis=0).T,self.p)
            

            #bcm rule square
            elif self.rule_p == 'bcm_sqr':
                gradient_p = np.outer(np.multiply(self.y, self.y) - np.multiply(self.y, bcm_threshold), self.previous_y) - self.p
            #bcm rule threshold 
            elif self.rule_p == 'bcm_thrs':
                gradient_p = np.outer(np.abs(self.y - bcm_threshold), self.previous_y) - self.p
            #stdp rule from biology and neuroscience
            elif self.rule_p == 'stdp':
                gradient_p = np.outer(self.y, self.previous_y) - np.outer(self.previous_y, self.y)
            elif self.rule_p == 'none':
                gradient_p = np.zeros((self.num_units_y, self.num_units_y))
            else:
                raise Exception("Rule P not implemented")
            
            gradient_list.append(gradient_p)

            if self.gate_p:
                gradient_p_T_modulated = np.multiply(gradient_p.T, spikes_y)
                gradient_p = gradient_p_T_modulated.T

            if self.dynamic_p or (self.rule_p == 'pehlevan'):
                self.p += (1/self.dynamic_learning_rate.reshape((self.num_units_y,1)))*gradient_p
            elif self.batched_p:
                if (self.iteration % self.batch_size) != 0:
                    self.accumulated_gradient_p += gradient_p
                else:
                    self.accumulated_gradient_p += gradient_p
                    self.p += self.lr_p*(self.accumulated_gradient_p/self.batch_size)
                    self.accumulated_gradient_p = np.zeros((self.num_units_y, self.num_units_y))
            else:
                self.p += self.lr_p*gradient_p
            
            if self.noisy_p:
                self.p += norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))*np.sqrt(self.lr_p)*self.noise_std_p
            
            if self.rectify_p:
                self.p[self.p < self.threshold_learning] = 0.0
            
            if self.zero_dig_p:
                np.fill_diagonal(self.p, 0.0)

            
            
            if self.kroned_p == 'yin':
                self.p = kron_matrix_yin(self.p, self.num_assemblies)
            elif self.kroned_p == 'yang':
                self.p = kron_matrix_yang(self.p, self.num_assemblies)
            elif self.kroned_p == 'chain':
                self.p = kron_matrix_chain(self.p, self.num_assemblies)
            elif self.kroned_p == 'cyclic':
                self.p = kron_matrix_cyclic_chain(self.p, self.num_assemblies)
            else:
                pass
        
        
        #Update M (z -> z)
        if "m" in active_matrices:
            if self.rule_m == 'hebbian': #delta mij = z_i * z_j - m_ij
                gradient_m = (np.outer(self.y, self.y)) - self.m
            elif self.rule_m == 'unc_hebbian': #delta mij = z_i * z_j - m_ij
                gradient_m = (np.outer(self.y, self.y))
            elif self.rule_m == 'symmetric':
                gradient_m = np.outer(self.y, self.y) - 0.5 * np.diag(np.multiply(self.y, self.y))
            elif self.rule_m == 'chklovskii': #delta mij = (z_i - z_mean) * (z_j - z_mean) - m_ij
                gradient_m = np.outer(self.y - y_mean, self.y - y_mean) - self.m
            elif self.rule_m == 'centered_hebbian': #delta mij = (|z_i - z_mean| - z_mean) * (|z_j - z_mean| - z_mean) - m_ij
                gradient_m = np.outer(self.y - self.bias_target, self.y - self.bias_target) - self.m
            elif self.rule_m == 'foldiak': #delta mij = (z_i * z_j) - (m_ij * z_i)
                gradient_m = np.outer(self.y, self.y) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.num_units_y, axis=0).T, self.m)
            elif self.rule_m == 'foldiak-constant': #delta mij = z_i * z_j - constant^2
                gradient_m = np.outer(self.y, self.y) - self.foldiak_constant**2
            elif self.rule_m == 'goodall':
                gradient_m = np.outer(self.y, self.y) - np.eye(self.num_units_y)
            elif self.rule_m == 'goodall_decay':
                gradient_m = np.outer(self.y, self.y) - np.eye(self.num_units_y) - self.m
            elif self.rule_m == 'goodall_chklovskii':
                gradient_m = np.outer(self.y - y_mean, self.y - y_mean) - np.eye(self.num_units_y)
            elif self.rule_m == 'goodall_chklovskii_decay':
                gradient_m = np.outer(self.y - y_mean, self.y - y_mean) - np.eye(self.num_units_y)/2 - self.m
            elif self.rule_m == 'goodall_chklovskii_exp':
                gradient_m = np.outer(self.y - self.y_exp_mean, self.y - self.y_exp_mean) - np.eye(self.num_units_y)
            elif self.rule_m == 'goodall_chklovskii_exp_decay':
                gradient_m = np.outer(self.y - self.y_exp_mean, self.y - self.y_exp_mean) - np.eye(self.num_units_y) - self.m
            elif self.rule_m == 'goodall_chklovskii_inverse':
                gradient_m = np.outer(self.y - self.y_exp_mean, self.y - self.y_exp_mean) - np.eye(self.num_units_y)
            elif self.rule_m == 'goodall_chklovskii_inverse_decay':
                gradient_m = np.outer(self.y - self.y_exp_mean, self.y - self.y_exp_mean) - np.eye(self.num_units_y)
            elif (self.rule_m == 'oja') or (self.rule_m == 'pehlevan'): #delta mij = z_i * (z_j - (m_ij * z_i)) = (z_i * z_j) - (m_ij * z_i^2))
                gradient_m = np.outer(self.y, self.y) - np.multiply(np.repeat(np.array([np.multiply(self.y, self.y)]), self.num_units_y, axis=0).T,self.m)
            elif self.rule_m == 'perceptron': #delta mij = (z_i - (m_ij * z_j)) * z_j   .... #does this makes sense??? 
                gradient_m = np.outer((self.y - (self.m @ self.y)), self.y)
            elif self.rule_m == 'balancer':
                gradient_m = np.outer(((self.w @ self.current_input) - self.y), self.y)
            elif self.rule_m == 'perceptron_nl': #delta mij = (z_i - (m_ij * z_j)) * z_j   .... #does this makes sense??? 
                gradient_m = np.outer(self.y - np.maximum((self.m @ self.y), self.threshold), self.y)
            elif self.rule_m == 'sfa':
                integrated_y = self.previous_y + self.y
                gradient_m = np.outer(integrated_y, integrated_y) - self.m
            elif self.rule_m == 'bcm_abs':
                gradient_m = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y) - self.m
            elif self.rule_m == 'bcm_abs_foldiak': #delta mij = (|y_i-y_i_mean| - y_i) * x_j) - m_ij
                gradient_m = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.num_units_y, axis=0).T,self.m)
            elif self.rule_m == 'pure_bcm_abs':
                gradient_m = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)
            else:
                raise Exception("Rule M not implemented")
            
            gradient_list.append(gradient_m)

            if self.gate_m:
                gradient_m_T_modulated = np.multiply(gradient_m.T, spikes_y)
                gradient_m = gradient_m_T_modulated.T

            if self.dynamic_m or (self.rule_m == 'pehlevan'):
                self.m += (1/self.dynamic_learning_rate.reshape((self.num_units_y,1)))*gradient_m
            elif self.rule_m == 'ff_outer':
                self.m = np.outer(self.w, self.w)
            elif self.batched_m:
                if (self.iteration % self.batch_size) != 0:
                    self.accumulated_gradient_m += gradient_m
                else:
                    self.accumulated_gradient_m += gradient_m
                    self.m += self.lr_m*(self.accumulated_gradient_m/self.batch_size)
                    self.accumulated_gradient_m = np.zeros((self.num_units_y, self.num_units_y))
            else:
                self.m +=self.lr_m*gradient_m


            if self.noisy_m:
                self.m += norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))*np.sqrt(self.lr_m)*self.noise_std_m
            

            if self.rectify_m:
                self.m[self.m < self.threshold_learning] = 0.0
            

            if self.zero_dig_m:
                np.fill_diagonal(self.m, 0.0)
            

            if self.kroned_m == 'yin':
                self.m = kron_matrix_yin(self.m, self.num_assemblies)
            elif self.kroned_m == 'yang':
                self.m = kron_matrix_yang(self.m, self.num_assemblies)
            else:
                pass

        self.normalize_weights(active_matrices)

        return gradient_list
    
    def normalize_weights(self, active_matrices):
        if self.rule_p == 'none':
            w_norm = np.copy(np.linalg.norm(self.w, axis=1, keepdims=True) + 1e-9)
            m_norm = np.copy(np.linalg.norm(self.m, axis=1, keepdims=True) + 1e-9)
            if ("w" in active_matrices) and self.normalize_w_weights:
                s = 0.1 / np.maximum(w_norm, 1e-12)
                self.w *= s
            if ("m" in active_matrices) and self.normalize_m_weights:
                s = 0.1 / np.maximum(m_norm, 1e-12)
                self.m *= s
        elif (self.rule_p == 'perceptron-single'):
            w_norm = np.copy(np.linalg.norm(self.w, axis=1, keepdims=True) + 1e-9)
            p_norm = np.copy(np.linalg.norm(self.p, axis=1, keepdims=True) + 1e-9)
            m_norm = np.copy(np.linalg.norm(self.m, axis=1, keepdims=True) + 1e-9)
            if ("w" in active_matrices) and self.normalize_w_weights:
                s = 0.1 / np.maximum(w_norm, 1e-12)
                self.w *= s
            if ("p" in active_matrices) and self.normalize_p_weights:
                s = 0.1 / np.maximum(p_norm, 1e-12)
                self.p *= s
            if ("m" in active_matrices) and self.normalize_m_weights:
                s = 0.1 / np.maximum(m_norm, 1e-12)
                self.m *= s
        else:
            excitatory_joint_matrix = np.concatenate([self.w, self.p], axis=1)
            excitatory_joint_row_norm = np.copy(np.linalg.norm(excitatory_joint_matrix, axis=1, keepdims=True) + 1e-9)
            inhibitory_norm = np.copy(np.linalg.norm(self.m, axis=1, keepdims=True) + 1e-9)
            if ("w" in active_matrices) and self.normalize_w_weights:
                s = 0.1 / np.maximum(excitatory_joint_row_norm, 1e-12)
                self.w *= s
            if ("p" in active_matrices) and self.normalize_p_weights:
                s = 0.1 / np.maximum(excitatory_joint_row_norm, 1e-12)
                self.p *= s
            if ("m" in active_matrices) and self.normalize_m_weights:
                s = 0.1 / np.maximum(inhibitory_norm, 1e-12)
                self.m *= s
    
    def update_parameters_sleep(self, active_matrices):
        #compute means
        if self.iteration > 0:
            y_mean = np.copy(self.track_y)/self.iteration
            y_squared_mean = np.copy(self.track_y_squared)/self.iteration
        else:
            y_mean = np.zeros(self.num_units_y)
            y_squared_mean = np.zeros(self.num_units_y)

        current_y = np.copy(self.y)

        #Compute BCM
        bcm_threshold = eval(self.bcm_trsh_exp)

        #Compute dynamic learning rates
        pehlevan_dynamic_lr = np.copy((self.track_y_squared*0.1)+self.dyn_lr_init)
        self.dynamic_learning_rate = eval(self.dynamic_lr_increment_exprss)
        
        gradient_list = list()

        
        #Update W (x -> y)
        if "w" in active_matrices:
            #correlation rule (classic hebbian rule)
            if self.rule_w == 'hebbian': #delta wij = (y_i * x_j) - w_ij
                gradient_w = np.outer(self.y, self.current_input) - self.w
            elif self.rule_w == 'bcm_abs': #delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - self.w
            else:
                gradient_w = np.zeros((self.num_units_y, self.inp_size))
            
            gradient_list.append(gradient_w)
            
            if self.gate_w:
                gradient_w_T_modulated = np.multiply(gradient_w.T, spikes_y)
                gradient_w = gradient_w_T_modulated.T

            if self.dynamic_w:
                self.w += (1/self.dynamic_learning_rate.reshape((self.num_units_y,1)))*gradient_w
            else:
                self.w += self.lr_w*gradient_w
            
            if self.noisy_w:
                self.w += norm.ppf(np.random.rand(self.num_units_y, self.inp_size))*np.sqrt(self.lr_w)*self.noise_std_w

            if self.rectify_w:
                self.w[self.w < self.threshold_learning] = 0.0
            
            if self.kroned_w == 'yin':
                self.w = kron_matrix_yin(self.w, self.num_assemblies)
            elif self.kroned_w == 'delayed':
                self.w = kron_matrix_yin(self.w, self.num_assemblies)
            elif self.kroned_w == 'yang':
                self.w = kron_matrix_yang(self.w, self.num_assemblies)
            else:
                pass
        
        
        #Update P (prev_y -> y)  
        if "p" in active_matrices:
            #correlation rule (classic hebbian rule)
            if self.rule_p == 'hebbian': #delta pij = (y_i * x_j) - p_ij
                gradient_p = np.outer(self.y, self.previous_y) - self.p
            elif self.rule_p == 'bcm_abs':
                gradient_p = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.previous_y) - self.p
            else:
                gradient_p = np.zeros((self.num_units_y, self.num_units_y))
            
            gradient_list.append(gradient_p)

            if self.dynamic_p:
                self.p += np.multiply((1/self.dynamic_learning_rate),gradient_p)
            else:
                self.p += self.lr_p*gradient_p
            
            if self.noisy_p:
                self.p += norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))*np.sqrt(self.lr_p)*self.noise_std_p
            
            if self.rectify_p:
                self.p[self.p < self.threshold_learning] = 0.0
            
            if self.zero_dig_p:
                np.fill_diagonal(self.p, 0.0)
            
            if self.kroned_p == 'yin':
                self.p = kron_matrix_yin(self.p, self.num_assemblies)
            elif self.kroned_p == 'yang':
                self.p = kron_matrix_yang(self.p, self.num_assemblies)
            elif self.kroned_p == 'chain':
                self.p = kron_matrix_chain(self.p, self.num_assemblies)
            elif self.kroned_p == 'cyclic':
                self.p = kron_matrix_cyclic_chain(self.p, self.num_assemblies)
            else:
                pass
        
        
        #Update M (z -> z)
        if "m" in active_matrices:
            if self.rule_m == 'hebbian': #delta mij = z_i * z_j - m_ij
                gradient_m = np.outer(self.y, self.y) - self.m
            else:
                gradient_m = np.zeros((self.num_units_y, self.num_units_y))
            
            gradient_list.append(gradient_m)

            if self.dynamic_m:
                self.m += (1/self.dynamic_learning_rate)*gradient_m
            else:
                self.m +=self.lr_m*gradient_m


            if self.noisy_m:
                self.m += norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))*np.sqrt(self.lr_m)*self.noise_std_m
            

            if self.rectify_m:
                self.m[self.m < self.threshold_learning] = 0.0
            

            if self.zero_dig_m:
                np.fill_diagonal(self.m, 0.0)
            

            if self.kroned_m == 'yin':
                self.m = kron_matrix_yin(self.m, self.num_assemblies)
            elif self.kroned_m == 'yang':
                self.m = kron_matrix_yang(self.m, self.num_assemblies)
            else:
                pass

        return gradient_list
    
    def get_name(self):
        return 'Lateral Inhibitory Layer (' + str(self.num_units_y) + ')'
    
    def get_hyperparams_dict(self):
        hparams = dict()
        hparams['name'] = self.name
        hparams['input_shape'] = self.inp_shape
        hparams['num_units_y'] = self.num_units_y
        hparams['num_blocks'] = self.num_assemblies
        hparams['step'] = self.step_size_y
        hparams['num_steps'] = self.num_steps
        hparams['multisteps'] = self.multisteps
        hparams['activation'] = self.activation
        hparams['threshold_dynamics'] = self.threshold
        hparams['spont_activity'] = self.spont_act
        hparams['bias'] = self.bias_expression
        hparams['modulator'] = self.modulate_stable_y_exprss
        hparams['threshold_learning'] = self.threshold_learning
        hparams['bcm_exprss'] = self.bcm_trsh_exp
        hparams['dynamic_lr_init'] = self.dyn_lr_init
        hparams['dynamic_lr_exprss'] = self.dynamic_lr_increment_exprss
        hparams['foldiak_constant'] = self.foldiak_constant
        hparams['zero_epoch'] = self.zero_epoch
        hparams['w_init'] = self.init_w
        hparams['w_update'] = self.rule_w
        hparams['w_lr'] = self.lr_w
        hparams['w_noisy'] = self.noisy_w
        hparams['w_dynamic'] = self.dynamic_w
        hparams['w_rectify'] = self.rectify_w
        hparams['w_kroned'] = self.kroned_w
        hparams['p_init'] = self.init_p
        hparams['p_update'] = self.rule_p
        hparams['p_lr'] = self.lr_p
        hparams['p_noisy'] = self.noisy_p
        hparams['p_dynamic'] = self.dynamic_p
        hparams['p_rectify'] = self.rectify_p
        hparams['p_kroned'] = self.kroned_p
        hparams['p_zero_dig'] = self.zero_dig_p
        hparams['m_init'] = self.init_m
        hparams['m_update'] = self.rule_m
        hparams['m_lr'] = self.lr_m
        hparams['m_noisy'] = self.noisy_m
        hparams['m_dynamic'] = self.dynamic_m
        hparams['m_rectify'] = self.rectify_m
        hparams['m_kroned'] = self.kroned_m
        hparams['m_zero_dig'] = self.zero_dig_m
        hparams['use_jax'] = self.use_jax
        hparams['sleep_sparsity'] = self.sleep_sparsity
        return hparams


#probs-async-rei-network
class ProbabilisticExcitatoryInhibitoryLayer:
    """

    """

    @staticmethod
    def default_hparams(input_shape):
        hparams = dict()
        hparams['input_shape'] = input_shape
        hparams['frequency'] = 10
        hparams['num_units_y'] = 500
        hparams['num_units_z'] = 100
        hparams['step_y'] = 0.01
        hparams['step_z'] = 0.01
        hparams['memory_size'] = 1000
        hparams['kernel'] = 'gaussian' # 'gaussian' or 'exponential' or 'uniform' or 'linear'
        hparams['kernel_param1'] = '0.5' #(mean) all functions are defined between 0 and 1
        hparams['kernel_param2'] = '0.1' #(std) only for gaussian
        hparams['kernel_amplitude'] = '1.0'
        hparams['spont_activity'] = 0.0
        hparams['threshold_dynamics'] = 0.01 # should be 0 for relu
        hparams['activation_y'] = 'relu'
        hparams['activation_z'] = 'relu'
        hparams['bias_y'] = 'zero_vector_y'
        hparams['bias_z'] = 'zero_vector_z'
        hparams['threshold_learning'] = 0.0
        hparams['voltage_mean'] = 1.4
        hparams['voltage_std'] = 0.4
        hparams['calcium_mean'] = 100
        hparams['calcium_std'] = 20
        hparams['w_init'] = 'random'
        hparams['w_seed'] = np.random.randint(0, 100000)
        hparams['w_update'] = 'bcm_abs'
        hparams['w_lr'] = 0.001
        hparams['w_rectify'] = True
        hparams['p_init'] = 'zero'
        hparams['p_seed'] = np.random.randint(0, 100000)
        hparams['p_update'] = 'bcm_abs'
        hparams['p_lr'] = 0.001
        hparams['p_rectify'] = True
        hparams['p_zero_dig'] = False
        hparams['m_init'] = 'random'
        hparams['m_seed'] = np.random.randint(0, 100000)
        hparams['m_update'] = 'hebbian'
        hparams['m_lr'] = 0.01
        hparams['m_rectify'] = True
        hparams['m_zero_dig'] = False
        hparams['u_init'] = 'random'
        hparams['u_seed'] = np.random.randint(0, 100000)
        hparams['u_update'] = 'hebbian'
        hparams['u_lr'] = 0.01
        hparams['u_rectify'] = True
        hparams['v_init'] = 'random'
        hparams['v_seed'] = np.random.randint(0, 100000)
        hparams['v_update'] = 'hebbian'
        hparams['v_lr'] = 0.01
        hparams['v_rectify'] = True
        return hparams


    #initialize from scratch (hparams, initial_params)
    def __init__(self, hparams):
        self.name = 'probs-async-rei-network'

        self.inp_shape = hparams['input_shape']
        self.inp_size = self.inp_shape[0]*self.inp_shape[1]*self.inp_shape[2]
        self.num_units_y = hparams['num_units_y']
        self.num_units_z = hparams['num_units_z']

        #ACTIVITY
        self.y = np.zeros(hparams['num_units_y']) #rate
        self.z = np.zeros(hparams['num_units_z']) #rate

        self.spikes_y = np.zeros(hparams['num_units_y'])
        self.spikes_z = np.zeros(hparams['num_units_z'])

        self.memory_size = hparams['memory_size']

        self.window_y = np.zeros((self.memory_size, self.num_units_y))
        self.window_z = np.zeros((self.memory_size, self.num_units_z))
        self.accumulated_y = np.zeros(self.num_units_y) # accumulated activity according to the kernel
        self.spike_trace_y = np.zeros((self.memory_size, self.num_units_y))
        self.spike_trace_z = np.zeros((self.memory_size, self.num_units_z))

        self.kernel = hparams['kernel']
        self.kernel_param1 = hparams['kernel_param1']
        self.kernel_param2 = hparams['kernel_param2']
        self.kernel_amplitude = hparams['kernel_amplitude']

        self.latest_gradient_y = np.zeros(hparams['num_units_y'])
        self.latest_gradient_z = np.zeros(hparams['num_units_z'])

        self.activation_y = hparams['activation_y']
        self.activation_z = hparams['activation_z']

        self.frequency = hparams['frequency']

        self.track_y = np.ones(hparams['num_units_y'])
        self.track_y_squared = np.ones(hparams['num_units_y'])
        self.track_spikes_y = np.ones(hparams['num_units_y'])
        self.track_z = np.ones(hparams['num_units_z'])
        self.track_z_squared = np.ones(hparams['num_units_z'])
        self.track_spikes_z = np.ones(hparams['num_units_z'])

        self.continuous_track_y = np.zeros(hparams['num_units_y'])
        self.continuous_track_z = np.zeros(hparams['num_units_z'])

        self.iteration = 1

        #MATRICES
        self.init_w = hparams['w_init']
        self.init_p = hparams['p_init']
        self.init_m = hparams['m_init']
        self.init_v = hparams['v_init']
        self.init_u = hparams['u_init']


        
        #feedforward matrix, x->y
        if self.init_w == 'random':
            self.w = np.random.rand(self.num_units_y, self.inp_size)
        elif self.init_w == 'zero':
            self.w = np.zeros((self.num_units_y, self.inp_size))
        else:
            raise Exception('No valid initial value for W provided')
        

        #lateral matrix, y->y
        if self.init_p == 'random':
            self.p = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_p == 'zero':
            self.p = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception('No valid initial value for F provided')
        
        

        
        #i to e matrix
        if self.init_u == 'random':
            self.u = np.random.rand(self.num_units_y, self.num_units_z)
        elif self.init_u == 'zero':
            self.u = np.zeros((self.num_units_y, self.num_units_z))
        elif self.init_u == 'identity':
            if self.num_units_y == self.num_units_z:
                self.u = np.identity(self.num_units_z)
            else:
                raise Exception('To be identity, len(x) = len(y)')
        elif self.init_u == 'posdef':
            if self.num_units_y == self.num_units_z:
                a = np.random.rand(self.num_units_z, self.num_units_z)
                self.u = a @ a.transpose()
                self.u = self.u / (np.max(self.u)+1)
            else:
                raise Exception('To be posdef, len(x) = len(y)')
        else:
            raise Exception('No valid initial value for U provided')
        
        

        #e to i matrix
        if self.init_v == 'random':
            self.v = np.random.rand(self.num_units_z, self.num_units_y)
        elif self.init_v == 'zero':
            self.v = np.zeros((self.num_units_z, self.num_units_y))
        elif self.init_v == 'identity':
            if self.num_units_y == self.num_units_z:
                self.v = np.identity(self.num_units_y)
            else:
                raise Exception('To be identity, len(x) = len(y)')
        elif self.init_v == 'posdef':
            if self.num_units_y == self.num_units_z:
                a = np.random.rand(self.num_units_z, self.num_units_z)
                self.v = a @ a.transpose()
                self.v = self.v / np.max(self.v)
            else:
                raise Exception('To be posdef, len(x) = len(y)')
        else:
            raise Exception('No valid initial value for V provided')
        


        #i to i matrix
        if self.init_m == 'random':
            self.m = np.random.rand(self.num_units_z, self.num_units_z)
        elif self.init_m == 'posdef':
            a = np.random.rand(self.num_units_z, self.num_units_z)
            self.m = a @ a.transpose()
        elif self.init_m == 'posdef_zerodig':
            a = np.random.rand(self.num_units_z, self.num_units_z)
            self.m = a @ a.transpose()
            np.fill_diagonal(self.z, 0.0)
        elif self.init_m == 'posdef_maxnorm':
            a = np.random.rand(self.num_units_z, self.num_units_z)
            self.m = a @ a.transpose()
            self.m = self.m / (np.max(self.m) + 1)
        elif self.init_m == 'posdef_maxnorm_zerodig':
            a = np.random.rand(self.num_units_z, self.num_units_z)
            self.m = a @ a.transpose()
            self.m = self.m / (np.max(self.m) + 1)
            np.fill_diagonal(self.m, 0.0)
        elif self.init_m == 'posdef_smooth_eigvals':
            a = np.random.rand(self.num_units_z, self.num_units_z)
            self.m = a @ a.transpose()
            self.m = self.m / np.max(self.m)
            D, U = eigh(self.m)
            D[0] = 1
            self.m = U @ np.diag(D) @ U.transpose()
        elif self.init_m == 'identity':
            self.m = np.identity(self.num_units_z)*0.3
        elif self.init_m == 'zero':
            self.m = np.zeros((self.num_units_z, self.num_units_z))
        else:
            raise Exception('No valid initial value for M provided')
        


        #DYNAMICS HYPERPARAMS

        #for the voltage dynamics
        self.threshold = hparams['threshold_dynamics']
        self.bias_expression_y = hparams['bias_y']
        self.bias_expression_z = hparams['bias_z']
        self.step_y = hparams['step_y']
        self.step_z = hparams['step_z']
        self.spont_act = hparams['spont_activity']

        #for the spiking dynamics

        self.voltage_mean_y = np.zeros(self.num_units_y)
        self.voltage_mean_y.fill(hparams['voltage_mean']) #mean of voltage threshold
        self.voltage_mean_z = np.zeros(self.num_units_z)
        self.voltage_mean_z.fill(hparams['voltage_mean']) #mean of voltage threshold
        self.voltage_std_y = np.zeros(self.num_units_y)
        self.voltage_std_y.fill(hparams['voltage_std']) #std of voltage threshold
        self.voltage_std_z = np.zeros(self.num_units_z)
        self.voltage_std_z.fill(hparams['voltage_std']) #std of voltage threshold

        self.calcium_mean_y = np.zeros(self.num_units_y)
        self.calcium_mean_y.fill(hparams['calcium_mean']) #mean of calcium concentration
        self.calcium_mean_z = np.zeros(self.num_units_z)
        self.calcium_mean_z.fill(hparams['calcium_mean']) #mean of calcium concentration
        self.calcium_std_y = np.zeros(self.num_units_y)
        self.calcium_std_y.fill(hparams['calcium_std']) #std of calcium concentration
        self.calcium_std_z = np.zeros(self.num_units_z)
        self.calcium_std_z.fill(hparams['calcium_std']) #std of calcium concentration

        self.time_elapsed_y = np.ones(hparams['num_units_y']) 
        self.time_elapsed_z = np.ones(hparams['num_units_z'])

        #LEARNING RATES
        self.threshold_learning = hparams['threshold_learning']

        self.lr_w = hparams['w_lr']
        self.lr_p = hparams['p_lr']
        self.lr_m = hparams['m_lr']
        self.lr_u = hparams['u_lr']
        self.lr_v = hparams['v_lr']

        self.rule_w = hparams['w_update']
        self.rule_p = hparams['p_update']
        self.rule_m = hparams['m_update']
        self.rule_u = hparams['u_update']
        self.rule_v = hparams['v_update']

        self.rectify_w = hparams['w_rectify']
        self.rectify_p = hparams['p_rectify']
        self.rectify_m = hparams['m_rectify']
        self.rectify_u = hparams['u_rectify']
        self.rectify_v = hparams['v_rectify']

        if 'm_zero_dig' in hparams.keys():
            self.zero_dig_m = hparams['m_zero_dig']
        else:
            self.zero_dig_m = False
        
        if 'p_zero_dig' in hparams.keys():
            self.zero_dig_p = hparams['p_zero_dig']
        else:
            self.zero_dig_p = False

        self.matrices = ['w', 'p', 'm', 'u', 'v']


    def set_learned_params(self, saved_w, saved_m, saved_p, saved_v, saved_u, saved_track_y, saved_num_spikes_y, saved_track_z, saved_num_spikes_z, saved_iterations):
        
        self.track_y = saved_track_y
        self.track_spikes_y = saved_num_spikes_y
        self.track_z = saved_track_z
        self.track_spikes_z = saved_num_spikes_z
        
        self.w = saved_w
        self.m = saved_m
        self.p = saved_p
        self.v = saved_v
        self.u = saved_u

        self.iteration = saved_iterations

    def roll_window(self):
        #roll window and add new activity
        self.window_y = np.roll(self.window_y, 1, axis=0)
        self.window_y[0] = self.y

        self.spike_trace_y = np.roll(self.spike_trace_y, 1, axis=0)
        self.spike_trace_y[0] = self.spikes_y

        x = np.linspace(0, 1, self.memory_size)
        if self.kernel == 'gaussian':
            if self.kernel_param1 != "*":
                kernel = float(self.kernel_amplitude)*np.exp(-np.square(x - float(self.kernel_param1)) / (2 * np.square(float(self.kernel_param2))))
                full_kernel = np.repeat(kernel, self.num_units_y)
            else:
                kernel_values = []
                for param in np.linspace(0.0, 1.0, self.num_units_y):
                    kernel = float(self.kernel_amplitude) * np.exp(-np.square(x - param) / (2 * np.square(float(self.kernel_param2))))
                    kernel_values.append(kernel)
                full_kernel = np.array(kernel_values)
        elif self.kernel == 'exponential':
            kernel = float(self.kernel_amplitude)*np.exp(-np.abs(x - float(self.kernel_param1)) / float(self.kernel_param2))
            full_kernel = np.repeat(kernel, self.num_units_y)
        elif self.kernel == 'laplace':
            kernel = float(self.kernel_amplitude)*np.exp(-np.abs(x - float(self.kernel_param1)) / float(self.kernel_param2)) / (2*float(self.kernel_param2))
            full_kernel = np.repeat(kernel, self.num_units_y)
        elif self.kernel == 'alpha':
            kernel = float(self.kernel_amplitude)*((x - float(self.kernel_param1))/float(self.kernel_param2)**2)*np.exp(-(x - float(self.kernel_param1))/float(self.kernel_param2))
            kernel[kernel < 0] = 0
            full_kernel = np.repeat(kernel, self.num_units_y)
        elif self.kernel == 'uniform':
            kernel = np.zeros(self.memory_size)
            kernel.fill(1.0/self.memory_size)
            kernel = float(self.kernel_amplitude)*kernel
            full_kernel = np.repeat(kernel, self.num_units_y)
        elif self.kernel == 'linear':
            kernel = float(self.kernel_amplitude)*np.zeros(self.memory_size)
            kernel.fill(1.0)
            full_kernel = np.repeat(kernel, self.num_units_y)
        else:
            raise Exception('No valid kernel provided')
        self.accumulated_y = np.sum(np.multiply(self.window_y, full_kernel.reshape(self.memory_size, self.num_units_y)), axis=0)

        return kernel
    
    #dynamics
    def gradient_step(self, current_input):
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)
        z_mean = np.divide(np.copy(self.track_z), self.track_spikes_z)
        z_squared_mean = np.divide(np.copy(self.track_z_squared), self.track_spikes_z)
        zero_vector_y = np.zeros(self.num_units_y)
        zero_vector_z = np.zeros(self.num_units_z)


        self.current_input = np.reshape(current_input, self.inp_size)

        #roll window and add new activity
        self.roll_window()
        
        gradient_y = (self.w @ self.current_input) + (self.p @ self.accumulated_y) - (self.u @ self.z) + (self.spont_act*np.random.uniform(low=0,high=1,size=(self.num_units_y))) - eval(self.bias_expression_y)
        self.y += self.step_y * gradient_y

        if self.activation_y == 'relu':
            self.y = np.maximum(self.y, self.threshold)
        elif self.activation_y == 'sigmoid':
            self.y = 1 / (1 + np.exp(-self.y))
        elif self.activation_y == 'linear':
            self.y = self.y
        elif self.activation_y == 'exp':
            self.y = np.exp(self.y)
        elif self.activation_y == 'tanh':
            self.y = np.tanh(self.y)
        else:
            raise Exception('No valid activation function provided')


        gradient_z = (self.v @ self.y) - (self.m @ self.z) + (self.spont_act*np.random.uniform(low=0,high=1,size=(self.num_units_z))) - eval(self.bias_expression_z)
        self.z += self.step_z * gradient_z

        if self.activation_z == 'relu':
            self.z = np.maximum(self.z, self.threshold)
        elif self.activation_z == 'sigmoid':
            self.z = 1 / (1 + np.exp(-self.z))
        elif self.activation_z == 'linear':
            self.z = self.z
        elif self.activation_z == 'exp':
            self.z = np.exp(self.z)
        elif self.activation_z == 'tanh':
            self.z = np.tanh(self.z)
        else:
            raise Exception('No valid activation function provided')
        
        self.continuous_track_y += self.y
        self.continuous_track_z += self.z

        self.iteration += 1

        self.latest_gradient_y = gradient_y
        self.latest_gradient_z = gradient_z

        return np.sum(np.abs(gradient_y)), np.sum(np.abs(gradient_z))
    

    def compute_spikes(self):
        # Standardize (vectorized)
        v_z_y = (self.y - self.voltage_mean_y) / self.voltage_std_y
        t_z_y = (self.time_elapsed_y - self.calcium_mean_y) / self.calcium_std_y

        v_z_z = (self.z - self.voltage_mean_z) / self.voltage_std_z
        t_z_z = (self.time_elapsed_z - self.calcium_mean_z) / self.calcium_std_z

        # Joint CDF for independent normals = product of univariate CDFs
        probs_y = norm.cdf(v_z_y) * norm.cdf(t_z_y)   # shape (n,)
        probs_z = norm.cdf(v_z_z) * norm.cdf(t_z_z)   # shape (n,)

        # Bernoulli sampling via RNG (returns boolean mask, no ints needed)
        rng = np.random.default_rng()
        mask_y = rng.random(probs_y.shape) < probs_y   # True where we "draw 1"
        self.spikes_y = mask_y.astype(dtype=np.float64)
        self.track_spikes_y += self.spikes_y

        # Zero in-place where sampled==1
        self.time_elapsed_y[mask_y] = 0.0
        self.time_elapsed_y += 1

        # Bernoulli sampling via RNG (returns boolean mask, no ints needed)
        rng = np.random.default_rng()
        mask_z = rng.random(probs_z.shape) < probs_z   # True where we "draw 1"
        self.spikes_z = mask_z.astype(dtype=np.float64)
        self.track_spikes_z += self.spikes_z

        # Zero in-place where sampled==1
        self.time_elapsed_z[mask_z] = 0.0
        self.time_elapsed_z += 1


    def projection(self):
        return np.reshape(self.w.transpose() @ self.y, self.inp_shape)

    # called at each beginning of sequence
    def reset_network(self):
        self.y = np.zeros(self.num_units_y)
        self.z = np.zeros(self.num_units_z)
        self.time_elapsed_y = np.ones(self.num_units_y)
        self.time_elapsed_z = np.ones(self.num_units_z)
        self.window_y = np.zeros((self.memory_size, self.num_units_y))
        self.window_z = np.zeros((self.memory_size, self.num_units_z))
        self.spike_trace_y = np.zeros((self.memory_size, self.num_units_y))
        self.spike_trace_z = np.zeros((self.memory_size, self.num_units_y))
    
    #plasticity
    def update_parameters(self):
        #compute spikes
        self.compute_spikes()

        #compute means
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)
        z_mean = np.divide(np.copy(self.track_z), self.track_spikes_z)
        z_squared_mean = np.divide(np.copy(self.track_z_squared), self.track_spikes_z)

        y_continuous_mean = self.continuous_track_y / self.iteration
        z_continuous_mean = self.continuous_track_z / self.iteration

        current_y = np.copy(self.y)
        current_z = np.copy(self.z)

        current_synaptic_input_y = (self.w @ self.current_input) + (self.p @ self.accumulated_y)

        modulated_activity_y = np.multiply(self.y, self.spikes_y)
        modulated_activity_z = np.multiply(self.z, self.spikes_z)

        modulated_mean_activity_y = np.multiply(y_mean, self.spikes_y)
        modulated_mean_activity_z = np.multiply(z_mean, self.spikes_z)

        #self.voltage_mean_y += float(self.vt_growth_rate)*self.spikes_y
        #self.voltage_mean_z += float(self.vt_growth_rate)*self.spikes_z

        #time_constant_potential = 0.001
        #self.voltage_thresholds_y += time_constant_potential*(modulated_mean_activity_y - self.voltage_thresholds_y)
        #self.voltage_thresholds_z += time_constant_potential*(modulated_mean_activity_z - self.voltage_thresholds_z)

        self.track_spikes_y += self.spikes_y
        self.track_y += modulated_activity_y
        self.track_y_squared += np.square(modulated_activity_y)
        self.track_spikes_z += self.spikes_z
        self.track_z += modulated_activity_z
        self.track_z_squared += np.square(modulated_activity_z)

        


        

        #Update M (z -> z)
        if self.rule_m == 'hebbian':
            gradient_m = np.outer(self.z, self.z) - self.m
            gradient_m_T_modulated = np.multiply(gradient_m.T, self.spikes_z) # modulate so only new postsynaptic neurons are affected
            gradient_m = gradient_m_T_modulated.T
        elif self.rule_m == 'hebbian_symmetric':
            gradient_m = np.outer(self.z, self.z) - self.m
            center_indices = np.where(self.spikes_z == 1)[0]
            modulated_matrix_cross = np.zeros((self.num_units_z, self.num_units_z))
            # Iterate through each center index and draw a cross
            for center_index in center_indices:
                # Horizontal line of the cross
                modulated_matrix_cross[center_index, :] = 1
                # Vertical line of the cross
                modulated_matrix_cross[:, center_index] = 1
            gradient_m = np.multiply(gradient_m, modulated_matrix_cross)
        else:
            gradient_m = np.zeros((self.num_units_z, self.num_units_z))
        

        self.m += self.lr_m * gradient_m
        
        if self.rectify_m:
            self.m[self.m < 0.0] = 0.0
        
        if self.zero_dig_m:
            np.fill_diagonal(self.m, 0.0)

        
        #Update F (y -> y)
        if self.rule_p == 'hebbian':
            gradient_p = np.outer(self.y, self.accumulated_y) - self.p
        elif self.rule_p == 'bcm_abs':
            bcm_thresh = y_mean/2
            gradient_p = np.outer((np.abs(self.y - bcm_thresh) - bcm_thresh), self.accumulated_y) - self.p
        elif self.rule_p == 'bcm_abs_continuous':
            bcm_thresh = y_continuous_mean/2
            gradient_p = np.outer((np.abs(self.y - bcm_thresh) - bcm_thresh), self.accumulated_y) - self.p
        elif self.rule_p == 'perceptron':
            gradient_p = np.outer((self.y - current_synaptic_input_y), self.accumulated_y)
        elif self.rule_p == 'perceptron_leaky':
            gradient_p = np.outer((self.y - (self.p@self.accumulated_y)), self.accumulated_y) - self.p
        else:
            gradient_p = np.zeros((self.num_units_y, self.num_units_y))
        
        gradient_p_T_modulated = np.multiply(gradient_p.T, self.spikes_y)
        gradient_p = gradient_p_T_modulated.T

        self.p += self.lr_p * gradient_p

        if self.rectify_p:
            self.p[self.p < 0.0] = 0.0
        
        if self.zero_dig_p:
            np.fill_diagonal(self.p, 0.0)
        

        #Update W (x -> y)
        if self.rule_w == 'hebbian':
            gradient_w = np.outer(self.y, self.current_input) - self.w
        elif self.rule_w == 'bcm_abs':
            bcm_thresh = y_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_thresh) - bcm_thresh), self.current_input) - self.w
        elif self.rule_w == 'bcm_abs_continuous':
            bcm_thresh = y_continuous_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_thresh) - bcm_thresh), self.current_input) - self.w
        elif self.rule_w == 'perceptron':
            gradient_w = np.outer((self.y - current_synaptic_input_y), self.current_input)
        elif self.rule_w == 'perceptron_leaky':
            gradient_w = np.outer((self.y - (self.w@self.current_input)), self.current_input) - self.w
        else:
            gradient_w = np.zeros((self.num_units_y, self.inp_size))

        gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
        gradient_w = gradient_w_T_modulated.T

        self.w += self.lr_w * gradient_w
        

        if self.rectify_w:
            self.w[self.w < 0.0] = 0.0

        
        #Update V (y -> z)
        if self.rule_v == 'hebbian':
            gradient_v = np.outer(self.z, self.y) - self.v
        elif self.rule_v == 'hebbian_symmetric':
            gradient_v = np.outer(self.z, self.y) - self.v
            center_indices_z = np.where(self.spikes_z == 1)[0]
            center_indices_y = np.where(self.spikes_y == 1)[0]
            modulated_matrix_cross = np.zeros((self.num_units_z, self.num_units_y))
            # Iterate through each center index and draw a cross
            for center_index in center_indices_z:
                # Horizontal line of the cross
                modulated_matrix_cross[center_index, :] = 1
            for center_index in center_indices_y:
                # Vertical line of the cross
                modulated_matrix_cross[:, center_index] = 1
            gradient_v = np.multiply(gradient_v, modulated_matrix_cross)
        elif self.rule_v == 'bcm_abs':
            bcm_thresh = z_mean/2
            gradient_v = np.outer((np.abs(self.z - bcm_thresh) - bcm_thresh), self.y) - self.v
        elif self.rule_v == 'bcm_abs_continuous':
            bcm_thresh = z_continuous_mean/2
            gradient_v = np.outer((np.abs(self.z - bcm_thresh) - bcm_thresh), self.y) - self.v
        else:
            gradient_v = np.zeros((self.num_units_z, self.num_units_y))
        
        gradient_v_T_modulated = np.multiply(gradient_v.T, self.spikes_z)
        gradient_v = gradient_v_T_modulated.T
        
        self.v += self.lr_v * gradient_v
        
        if self.rectify_v:
            self.v[self.v < 0.0] = 0.0


        #Update U (z -> y)
        if self.rule_u == 'hebbian':
            gradient_u = np.outer(self.y, self.z) - self.u
        elif self.rule_u == 'hebbian_symmetric':
            gradient_u = np.outer(self.y, self.z) - self.u
            center_indices_y = np.where(self.spikes_y == 1)[0]
            center_indices_z = np.where(self.spikes_z == 1)[0]
            modulated_matrix_cross = np.zeros((self.num_units_y, self.num_units_z))
            # Iterate through each center index and draw a cross
            for center_index in center_indices_y:
                # Horizontal line of the cross
                modulated_matrix_cross[center_index, :] = 1
            for center_index in center_indices_z:
                # Vertical line of the cross
                modulated_matrix_cross[:, center_index] = 1
            gradient_u = np.multiply(gradient_u, modulated_matrix_cross)
        else:
            gradient_u = np.zeros((self.num_units_y, self.num_units_z))
        
        gradient_u_T_modulated = np.multiply(gradient_u.T, self.spikes_y)
        gradient_u = gradient_u_T_modulated.T

        self.u += self.lr_u * gradient_u
        
        if self.rectify_u:
            self.u[self.u < 0.0] = 0.0


    def get_name(self):
        return 'Probabilistic EI Network (' + str(self.num_units_y) + ',' + str(self.num_units_z) + ')'


    def get_extensive_name(self):
        net_name = "Probabilistic EI Network (" + str(self.num_units_y) + ',' + str(self.num_units_z) + ")  -  "
        net_name += "W: " + self.init_w + " -> " + self.rule_w + " ... "
        net_name += "M: " + self.init_m + " -> " + self.rule_m + " ... "
        net_name += "V: " + self.init_v + " -> " + self.rule_v + " ... "
        net_name += "U: " + self.init_u + " -> " + self.rule_u + " ... "

        return net_name

    
    def get_hyper_parameters_string(self):
        w_params = 'W init: ' + self.init_w + 'rule: ' + self.rule_w + ' lr: ' + str(self.lr_w) + ' dyn: ' + ' rect: ' + str(self.rectify_w) + '  \n '
        m_params = 'M init: ' + self.init_m + 'rule: ' + self.rule_m + ' lr: ' + str(self.lr_m) + ' dyn: ' + ' rect: ' + str(self.rectify_m) + '  \n '
        u_params = 'U init: ' + self.init_u + 'rule: ' + self.rule_u + ' lr: ' + str(self.lr_u) + ' dyn: ' + ' rect: ' + str(self.rectify_u) + '  \n '
        v_params = 'V init: ' + self.init_v + 'rule: ' + self.rule_v + ' lr: ' + str(self.lr_v) + ' dyn: ' + ' rect: ' + str(self.rectify_v) + '  \n '
        
        all_params = 'y_units: ' + str(self.num_units_y) + '   z_units: ' + str(self.num_units_z) + ' \n'
        all_params += ' step_y: ' + str(self.step_size) + ' \n'
        all_params += 'spontaneous__activity: ' + str(self.spont_act) + ' \n'
        all_params += 'global inhibit at each sequence: ' + str(self.zero_begin_seq) + ' \n'
        all_params += 'bcm_update_y: ' + self.bcm_trsh_exp_y + '   bcm_update_z: ' + self.bcm_trsh_exp_z + ' \n '

        all_params += 'threshold: ' + str(self.threshold) + ' \n '
        all_params += 'bias_y: ' + str(self.bias_expression_y) + '  bias_z: ' + str(self.bias_expression_z) + ' \n '
        all_params += w_params + m_params + u_params + v_params +' \n '

        return all_params


    def get_hyperparams_dict(self):
        hparams = dict()
        hparams['name'] = self.name
        hparams['input_shape'] = self.inp_shape
        hparams['frequency'] = self.frequency
        hparams['num_units_y'] = self.num_units_y
        hparams['num_units_z'] = self.num_units_z
        hparams['step_y'] = self.step_y
        hparams['step_z'] = self.step_z
        hparams['memory_size'] = self.memory_size
        hparams['kernel'] = self.kernel
        hparams['kernel_param1'] = self.kernel_param1
        hparams['kernel_param2'] = self.kernel_param2
        hparams['spont_activity'] = self.spont_act
        hparams['threshold_dynamics'] = self.threshold
        hparams['activation_y'] = self.activation_y
        hparams['activation_z'] = self.activation_z
        hparams['bias_y'] = self.bias_expression_y
        hparams['bias_z'] = self.bias_expression_z
        hparams['threshold_learning'] = self.threshold_learning
        hparams['voltage_threshold_y'] = self.voltage_thresholds_y
        hparams['voltage_threshold_z'] = self.voltage_thresholds_z
        hparams['calcium_growth_expression_y'] = self.calcium_growth_expression_y
        hparams['calcium_growth_expression_z'] = self.calcium_growth_expression_z
        hparams['depresion_expression_y'] = self.depression_expression_y
        hparams['depresion_expression_z'] = self.depression_expression_z
        hparams['w_init'] = self.init_w
        hparams['w_update'] = self.rule_w
        hparams['w_lr'] = self.lr_w
        hparams['w_rectify'] = self.rectify_w
        hparams['p_init'] = self.init_p
        hparams['p_update'] = self.rule_p
        hparams['p_lr'] = self.lr_p
        hparams['p_rectify'] = self.rectify_p
        if hasattr(self, 'zero_dig_p'):
            hparams['p_zero_dig'] = self.zero_dig_p
        if hasattr(self, 'zero_dig_m'):
            hparams['m_zero_dig'] = self.zero_dig_m
        hparams['m_init'] = self.init_m
        hparams['m_update'] = self.rule_m
        hparams['m_lr'] = self.lr_m
        hparams['m_rectify'] = self.rectify_m
        hparams['u_init'] = self.init_u
        hparams['u_update'] = self.rule_u
        hparams['u_lr'] = self.lr_u
        hparams['u_rectify'] = self.rectify_u
        hparams['v_init'] = self.init_v
        hparams['v_update'] = self.rule_v
        hparams['v_lr'] = self.lr_v
        hparams['v_rectify'] = self.rectify_v
        return hparams

