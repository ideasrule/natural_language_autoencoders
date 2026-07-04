The current code was written to train a natural language autoencoder.  Please read https://transformer-circuits.pub/2026/nla to learn what a NLA is.  Then, please read train_qwen3.5.sh to learn how to run the code to train a NLA for Qwen3.5-9B, and setup_qwen3.5.sh to learn how to set up the environment.

Your task is to adapt the code for Qwen3.6-27B.  The architectures are very similar, so it's possible the code will work out of the box.  After adapting the code, please smoke test the AR SFT, AV SFT, and RL stages, then start a full training run.  The data is in /workspace/data/.

Remember that this machine only has 1000 GB of disk space on /workspace and much less on the home directory, so please prune checkpoints as needed.

