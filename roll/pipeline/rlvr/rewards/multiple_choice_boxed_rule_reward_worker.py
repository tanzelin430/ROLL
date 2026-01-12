# 导入必要的库和模块
from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy

from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider

from typing import Union, Dict, List

from roll.utils.logging import get_logger
from roll.pipeline.rlvr.rewards.utils import extract_option_letter

logger = get_logger()  # 获取日志记录器实例


def multiple_choice_boxed_reward_fn(response: str, ground_truth: str, valid_options: str = 'ABCDEFGHIJ'):
    """
    Extract and verify multiple choice answer from response.

    Supports options A-J (configurable via valid_options).
    Uses robust extraction with multiple strategies:
    1. \\boxed{} content
    2. "Final Answer:" pattern
    3. "The answer is:" pattern
    4. Last valid option letter in text

    Args:
        response: Model response text
        ground_truth: Expected answer (e.g., "A", "B", etc.)
        valid_options: String of valid option letters (default: A-J)

    Returns:
        Tuple of (extracted_answer, reward, format_flag, correct_flag)
    """
    # Use robust extraction from utils
    extracted_answer = extract_option_letter(response, valid_options)

    # Check if answer was found in \boxed{}
    format_flag = '\\boxed{' in response and extracted_answer is not None

    # Check correctness
    correct_flag = False
    if extracted_answer is not None and ground_truth:
        # Compare first character (ground_truth might be "A" or "A.")
        gt_letter = ground_truth[0].upper() if ground_truth else None
        correct_flag = (extracted_answer == gt_letter)

    # Compute reward: require both format and correctness for full reward
    if correct_flag and format_flag:
        reward = 1.0
    else:
        reward = 0.0

    return extracted_answer or "", reward, format_flag, correct_flag


class MultipleChoiceBoxedRuleRewardWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        batch_size = len(response_text_list)

        # Decode prompts from tensor (same as MathRuleRewardWorker)
        prompts = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=False)
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch["tag"]

        multiple_choice_boxed_rewards = []
        scores = []

        for i, (resp_tokens, ground_truth, tag, prompt) in enumerate(
            zip(data.batch["responses"], ground_truths, tags, prompts)
        ):
            ori_resp_text = self.tokenizer.decode(resp_tokens, skip_special_tokens=False)
            resp_text_without_sptoken = (
                ori_resp_text.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
            )

            # extract_option_letter handles both with/without </think> tags
            extracted_answer, multiple_choice_boxed_reward, format_flag, correct_flag = multiple_choice_boxed_reward_fn(
                resp_text_without_sptoken, ground_truth
            )

            # score: 1 for correct, 0 for incorrect
            score = 1.0 if correct_flag else 0.0

            # 存到 multiple_choice_boxed_rewards
            multiple_choice_boxed_rewards.append(multiple_choice_boxed_reward)
            scores.append(score)

            try:
                outputs = json.dumps(
                    {
                        "multiple_choice_boxed_reward": multiple_choice_boxed_reward,
                        "format_flag": format_flag,
                        "correct_flag": correct_flag,
                        "prompt": str(prompt),
                        "extracted_answer": str(extracted_answer),
                        "ground_truth": str(ground_truth),
                        "response": str(resp_text_without_sptoken),
                    },
                    ensure_ascii=False,
                )
                self.logger.debug(outputs)
            except Exception as e:
                self.logger.error(f"answer check except: {e}")


        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        scores = torch.tensor(scores, dtype=torch.float16)
        response_level_rewards = torch.zeros_like(scores, dtype=torch.float16)
        # 5) 将这些张量打包进同一个字典
        # TODO: 不同的reward worker的output是否需要统一output，或者有没有自适应的办法，避免在新增监控量时每个worker都需要修改
        output_tensors = {
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": response_level_rewards,
            "scores": scores,
        }

        # 6) 用 DataProto.from_dict(...) 构造返回值
        output = DataProto.from_dict(tensors=output_tensors)
        return output
