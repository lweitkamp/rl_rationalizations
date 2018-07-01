# Explainable Reinforcement Learning: Visual Policy Rationalizations Using Grad-CAM
This reposity is holds the code needed to reproduce the results for my thesis.
The Grad-CAM implementaion used is based on https://github.com/kazuto1011/grad-cam-pytorch,
the A3C implementation is based on https://github.com/ikostrikov/pytorch-a3c.

## How to use the code

python3 main.py --env-name ENV_NAME --gradcam_layer GCAM_LAYER

This command should run an episode of ENV_NAME using the pretrained models and collect Grad-CAM outputs
for each state/action at layer GCAM_LAYER (which would default to features.elu4).


## trained models
The models can be found in the pretrained folder. Each model has both a 'full' and a 'half' version (see paper).

|                    | Full Agent Mean | Full Agent Variance | Half Agent Mean | Half Agent Variance | DeepMind |
|--------------------|:---------------:|---------------------|-----------------|---------------------|----------|
| \textbf{Pong}      |           21.00 | 0.00                | 14.99           | 0.09                |          |
| \textbf{BeamRider} |         4659.04 | 1932.58             | 1597.40         | 1202.00             |          |
| \textbf{Seaquest}  |         1749.00 | 11.44               | N/A             | N/A                 |          |


Note that the DeepMind results are based on time interval (24 hours) and not on amount of frames used.
There is also probably a difference in environment used which could impact the return.

