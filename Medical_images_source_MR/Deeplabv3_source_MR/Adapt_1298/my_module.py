import os
import numpy as np

def process_gmm(args):
    
    gmm, idx, curr_data = args
    try:
        log_likelihood = gmm.score_samples(curr_data)
        return log_likelihood.mean(), idx
    except Exception as e:
        print(f"Error in process_gmm: {e}")
        return float('-inf'), idx