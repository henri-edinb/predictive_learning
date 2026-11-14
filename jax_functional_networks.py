import jax
import jax.numpy as jnp
import jax.scipy as jsp
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import datetime
import math


from jax import config
config.update("jax_enable_x64", True)
import optax

import jax.tree_util as jtu
from sklearn.metrics.pairwise import cosine_similarity
from skimage.util import view_as_blocks, view_as_windows


#one step of fast dynamics
def lil_dynamics_step_relu(current_input, current_y_feedback, current_y, w, m, p, step, thrsh, bias_y):
    gradient_y = (w @ current_input) + (p @ current_y_feedback) - (m @ current_y) - bias_y
    new_y = current_y + (step * gradient_y)
    new_y = jnp.maximum(new_y, thrsh)
    
    return new_y, gradient_y

#runs n steps of fast dyanmics
def lil_dynamics_relu(current_input, current_y_feedback, current_y, w, m, p, step, thrsh, num_steps, bias_y):
    gradient_y_norm = jnp.zeros(num_steps, dtype=jnp.float64)
    entropy_y = jnp.zeros(num_steps, dtype=jnp.float64)
    
    new_y = current_y
    
    for euler_step in range(0, num_steps):
        old_y = new_y
        new_y, grad_y = lil_dynamics_step_relu(current_input, current_y_feedback, old_y, w, m, p, step, thrsh, bias_y)
        gradient_y_norm = gradient_y_norm.at[euler_step].set(jnp.linalg.norm(grad_y))
        entropy_y = entropy_y.at[euler_step].set(-jnp.sum(new_y*jnp.log(new_y)) / jnp.log(2))

    return new_y, gradient_y_norm, entropy_y

lil_dynamics_relu_jit = jax.jit(lil_dynamics_relu, static_argnums=(8,))