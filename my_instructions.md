The current code was written to train a natural language autoencoder.  Please read https://transformer-circuits.pub/2026/nla to learn what a NLA is.  Then, please read train_qwen2.5.sh to learn how to run the code to train a NLA for Qwen2.5-7B, and README.md to learn how to set up the environment.

Your task is to adapt the code for Qwen3.5-9B.  I recommend the following steps:

1. Pull the latest versions of sglang and miles from GitHub.  These should already support Qwen3.5.
2. Modify the patches in patches/, then apply them to the sglang and miles codes you just pulled.
3. Install sglang, miles, the latest version of transformers (not 4.57), nla, and anything else needed to train a NLA for Qwen3.5-9B.
4. Train a NLA for Qwen3.5-9B, including the AR SFT, AV SFT, and RL stages.  The data is in /workspace/data.  During all of training, please disable thinking.

Remember that this box only has 1000 GB of disk space on /workspace and much less on the home directory, so please prune checkpoints as needed.