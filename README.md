# LLM-based Few-Shot Early Rumor Detection with Imitation Agent

This is the official repository of an agile and cost-effective framework for few-shot early rumor detection, which comprises a lightweight agent for automatic early time point determination and an LLM for rumor detection. This work has been accepted at KDD 2026.

<p align="center">
<img src="misc/main_fig.png" height=350>
</p>

## Requirements
Our implementation is built upon the [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) codebase. Please refer to the original repository for detailed setup instructions.

## Data Format
Please ensure that all data is organized in the following structure and saved in JSON format::
 ```
  {
  "instance_id": {
          "label": "rumor",
          "seq": { # sorted by timestamp
              "timeline": [timestamp,timestamp,...],
              "tids": [post_id,post_id,...],
              'texts': [post, post, ...]
              }}
  ...
  }
  ```

## Prepare Expert Trajectories
Please place all data files (including expert trajectories, env trajectories and inference data) under the `data` folder, and run `gen_exp_trajs.py`.

## Training & Inference
Once the data is prepared, run `main.py`.

## Citation

```
@misc{zeng2025llmbasedfewshotearlyrumor,
      title={LLM-based Few-Shot Early Rumor Detection with Imitation Agent}, 
      author={Fengzhu Zeng and Qian Shao and Ling Cheng and Wei Gao and Shih-Fen Cheng and Jing Ma and Cheng Niu},
      year={2025},
      eprint={2512.18352},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.18352}, 
}

```
