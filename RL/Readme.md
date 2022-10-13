### How to train the reinforcement learning agent
Tested with python 3.9
First install requirements.txt

Download the semantic segmentation model from [here](https://drive.google.com/file/d/1JRdPggs5jTWAXKRXk6hVxzmP-KnOr8Hw/view?usp=sharing) and place it inside Segmentation_model

Then, you can run the mario_ddqn.py file, it has multiple options. The command to train on level 1-1 using semantic segmentation, and saving weights on 1_1_ssweights is:

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -t

If you want to visualize it, use -vis, we recommend only visualizing on inference.

Once it trained, if you want to see how the model behaves and plays, run:

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -pt -vis

You can also change the level with --level

    python .\mario_ddqn.py -wd 1_1_ssweights -it ss -t --level 2-1

For a full list of commands do:

    python .\mario_ddqn.py -h


