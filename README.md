# Explainable Reinforcement Learning: Visual Policy Rationalizations Using Grad-CAM
This reposity is holds the code needed to reproduce the results for my thesis.
The Grad-CAM implementation used is based on https://github.com/kazuto1011/grad-cam-pytorch,
the A3C implementation is based on https://github.com/ikostrikov/pytorch-a3c.

## How to use the code
Note that it is still a work in progress, but eventually it should look like this:

```
python3 main.py --env-name ENV_NAME --gradcam_layer GCAM_LAYER
```

This command should run an episode of ENV_NAME using the pretrained models and collect Grad-CAM outputs
for each state/action at layer GCAM_LAYER (which would default to features.elu4).


## trained models
The models can be found in the pretrained folder. Each model has both a 'full' and a 'half' version (see paper).

|                    | Full Agent Mean | Full Agent Variance | Half Agent Mean | Half Agent Variance | DeepMind |
|--------------------|:---------------:|---------------------|-----------------|---------------------|----------|
| *Pong*      |           21.00 | 0.00                | 14.99           | 0.09                | 10.7         |
| *BeamRider* |         4659.04 | 1932.58             | 1597.40         | 1202.00             | 24622.2         |
| *Seaquest*  |         1749.00 | 11.44               | N/A             | N/A                 | 1326.1         |


Note that these models are based on amount of frames whereas DeepMind is based on 4 day training on 16 CPU cores, which
makes comparing them hard.
