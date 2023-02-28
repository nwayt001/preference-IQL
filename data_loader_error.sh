#!/bin/bash

# Trains OpenAI VPT models using IQ-Learn algorithm
python iq_learn/Minimal_example_data_loader_error.py agent=sac method=iq env=basalt_findcave agent.init_temp=0.001 method.chi=True
