import logging

# WARNING:latex2sympy2_extended.math_normalization:equations is deprecated, as it handled by the parser now
logging.getLogger('latex2sympy2_extended.math_normalization').setLevel(logging.ERROR)

from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from math_verify import parse, verify
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider
from roll.utils.context_managers import state_offload_manger
from roll.pipeline.rlvr.rewards.utils import extract_last_boxed

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _hf_verify_math_sample(response, answer, result, prompt):
    try:
        # 使用健壮的 \boxed{} 提取，不依赖 \n\n 或 </think> 分隔符
        # 这在 enable_thinking=False 时也能正常工作
        boxed_content = extract_last_boxed(response)

        # 如果找到 \boxed{}，直接解析其内容
        # 否则尝试让 parse() 从完整响应中提取
        if boxed_content is not None:
            # 将提取的内容包装为 \boxed{} 格式，让 parse() 处理
            cleaned_response = f"\\boxed{{{boxed_content}}}"
        else:
            # 没有找到 \boxed{}，尝试用完整响应
            cleaned_response = response

        parsed_answers = parse(cleaned_response, fallback_mode="no_fallback")

        # 如果解析结果为空，则认为提取失败
        if not parsed_answers:
            exect_answer = None
        else:
            # 通常我们只关心第一个解析出的结果
            exect_answer = parsed_answers[0]

        gold_answer = parse(answer)

        if gold_answer is None or exect_answer is None:
            result.append((False, "", ""))
        else:
            # 假设 verify 函数可以处理 parse 返回的对象
            ans = verify(gold_answer[0], exect_answer)
            result.append((ans, str(gold_answer[0]), str(exect_answer)))

    except Exception as e:
        # 捕获任何潜在的异常，确保进程不会崩溃
        result.append((False, "", ""))


def hf_verify_math_sample(answer_a, answer_b, prompt, timeout_sec=5.0):
    with multiprocessing.Manager() as manager:
        result = manager.list()
        
        p = multiprocessing.Process(
            target=_hf_verify_math_sample,
            args=(answer_a, answer_b, result, prompt)
        )
        
        p.start()
        try:
            max_timeout = min(timeout_sec + 1, 10)
            p.join(timeout=max_timeout)
        except Exception as e:
            pass
        finally:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
            p.join(timeout=2)
        if not result:
            return False, "", ""
        return result[0]

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    def repetition_penalty_reward(response, **kwargs) -> float:
        if response == "" or len(response.split()) < ngram_size:
            return 0.0
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1
        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        return reward
    return repetition_penalty_reward

def long_block_penalty_reward_fn(text: str, max_length: int = 100) -> float:
    max_block_len = max([len(i) for i in text.split(" ")])
    reward = -float(max_block_len > max_length)
    return reward

def format_reward_fn(text: str, pattern: Optional[str] = r"^<think>.*?</think>.*?<answer>.*?</answer>$"):
    if pattern is None:
        pattern: str = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matche = re.match(pattern, text, re.DOTALL | re.MULTILINE)
    reward = 0 if matche else -1
    return reward


class MathRuleRewardWorker(Worker):
    """
    (x)Reward Model 使用 AutoModelForSequenceClassification 协议
    面向math的rule reward model
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.1)
        self.format_pattern = getattr(self.worker_config, "format_pattern", None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        verify_answer = []
        repetition_penalty_rewards = []
        long_block_penalty_rewards = []
        response_length_rewards = []
        format_rewards = []

        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        prompt_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=False)

        # Check if ground_truth exists
        if "ground_truth" not in data.non_tensor_batch:
            self.logger.error("ground_truth not found in non_tensor_batch!")
            # Return zeros
            batch_size = len(response_text_list)
            token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
            scores = torch.zeros(batch_size, dtype=torch.float16)
            return DataProto.from_dict(
                tensors={
                    "token_level_rewards": token_level_rewards,
                    "response_level_rewards": scores,
                    "scores": scores,
                }
            )

        for response, answer, prompt in zip(response_text_list, data.non_tensor_batch["ground_truth"], prompt_text_list):
            
            prompt = prompt.replace("<|endoftext|>", "").replace("<pad>", "")
            response = response.replace("<|endoftext|>", "").replace("<pad>", "")
            # self.logger.info(json.dumps({
            #     "prompt": prompt}, ensure_ascii=False))
            
            try:
                with timeout(5):
                    correct, extracted_ground_truth, extracted_response = hf_verify_math_sample(
                        response, f"${answer}$", prompt
                    )

                log_data = {
                    "response": response[-200:],  # Last 200 chars
                    "extracted_response": extracted_response,
                    "answer": answer,
                    "extracted_ground_truth": extracted_ground_truth,
                    "correct": correct,
                }
                self.logger.debug(json.dumps(log_data, ensure_ascii=False))

            except Exception as e:
                self.logger.error(f"timeout or error during hf_verify_math_sample. answer: {answer}, response: {response}")
                correct = False
                extracted_response = ""
                extracted_ground_truth = ""
            
            if correct:
                verify_answer.append(1)
            else:
                verify_answer.append(0)
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(response))
            format_rewards.append(format_reward_fn(response, self.format_pattern))
            long_block_penalty_rewards.append(long_block_penalty_reward_fn(response))
            response_length_rewards.append(len(response) / 20000)
            
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_length_rewards = torch.tensor(response_length_rewards, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        long_block_penalty_rewards = torch.tensor(long_block_penalty_rewards, dtype=torch.float16)
        format_rewards = torch.tensor(format_rewards, dtype=torch.float16)
        scores = torch.tensor(verify_answer, dtype=torch.float16)
        response_level_rewards = torch.tensor(verify_answer, dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
            }
        )

        self.logger.debug(f"reward output: {output}, response_level_rewards: {response_level_rewards}")
        return output