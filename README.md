# Weakly-Supervised Moment Retrieval Network for Video Corpus Moment Retrieval

Code for the paper **WMRN: Weakly-Supervised Moment Retrieval Network for Video Corpus Moment Retrieval**, ICIP 2021.

**Author**: Sunjae Yoon, Dahyun Kim, Ji Woo Hong, Junyeong Kim, Kookhoi Kim, Chang D. Yoo

This work was partly supported by Institute for Information communications Technology Planning Evaluation(IITP) grant funded by the Korea government(MSIT) (2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by LIG-Nex1 Co. through grant Y20-005



## Installation

## Requirements 
- [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+),
- [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+),
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

We build our model on top of [HERO](https://github.com/linjieli222/HERO) and [TVR](https://github.com/jayleicn/TVRetrieval).
It is compatible with the requirements and quick start of HERO, if you have any problem please refer above link also.

## Quick Start
1. Pretrained model HERO: Run `bash scripts/download_pretrained.sh $PATH_TO_STORAGE` to get latest pretrained
checkpoints. We use the HowTo100M pre-tasks pretrained model in HERO.


2. Load dataset
    ```bash
    bash scripts/download_tvr.sh $PATH_TO_STORAGE
    ```

3. We utilize the Docker from HERO, which gives command for pooling docker image below.
    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/video_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```


3. Train weakly-supervised video corpus moment retrieval
    ```bash
    # inside the container
    horovodrun -np 8 python train.py --config config/train-tvr-8gpu.json
    ```


4. Eval weakly-supervised video corpus moment retrieval
    ```bash
    # inside the container
    horovodrun -np 8 python eval.py --query_txt_db /txt/tvr_val.db/ --split val \
        --vfeat_db /video/tv/ --sub_txt_db /txt/tv_subtitles.db/ \
        --output_dir /storage/tvr_default/ --checkpoint 4800 --fp16 --pin_mem

    ```
    The result file will be written at `/storage/tvr_default/results_val/results_4800_all.json`.
    Change to  ``--query_txt_db /txt/tvr_test_public.db/ --split test_public`` for inference on test_public split.
    Please format the result file as requested by the evaluation server for submission, our code does not include formatting.

## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{yoon2021weakly,
  title={Weakly-Supervised Moment Retrieval Network for Video Corpus Moment Retrieval},
  author={Yoon, Sunjae and Kim, Dahyun and Hong, Ji Woo and Kim, Junyeong and Kim, Kookhoi and Yoo, Chang D},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={534--538},
  year={2021},
  organization={IEEE}
}
```

## License

MIT









































