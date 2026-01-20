"""
DeepSeek API Reward Worker for mathematical proof evaluation.

This worker uses DeepSeek API (or compatible OpenAI-style API) as an oracle verifier
to score mathematical proofs. No local GPU required - all inference happens remotely.

Architecture:
- Workers send HTTP POST requests to remote API (e.g., DeepSeek V3.2)
- Multiple workers can send concurrent requests (limited by API rate limits)
- All GPU resources can be used for actor training and inference

Usage:
Configure in yaml:
  rewards:
    proof_gen:
      worker_cls: roll.pipeline.rlvr.rewards.deepseek_api_reward_worker.DeepSeekAPIRewardWorker
      world_size: 16  # HTTP clients, no GPU needed
      device_mapping: []  # No GPU required
      judge_model_type: api
      judge_api_url: "http://your-api-endpoint/v1"
      judge_api_key: "your-api-key"
      judge_model_name: "deepseek-ai/DeepSeek-V3.2"
"""
import os
import re
import time
from typing import Any, Dict, Optional

import requests
import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.rlvr.rewards.utils import extract_last_boxed


def extract_solution_after_think(response: str) -> str:
    """Extract the solution part after </think> tag."""
    think_end_tag = "</think>"
    last_think_end = response.rfind(think_end_tag)

    if last_think_end != -1:
        solution = response[last_think_end + len(think_end_tag):].strip()
        if not solution:
            return response
        return solution
    return response


# Verifier prompt template (same as oracle labeling)
PROMPT_VERIFICATION = """## Instruction
Your task is to evaluate the quality of a solution to a mathematical problem. The problem may ask for a proof or an answer. If an answer is required, the solution should provide both the answer and a valid justification.

Please evaluate the solution according to the following criteria:

**Score 1 (Correct):** The solution correctly solves the problem. The core reasoning is sound, key steps are present, and the conclusion is valid. Minor omissions of "obvious" or well-known results (e.g., basic inequalities, standard theorems) are acceptable as long as the logical flow is clear.

**Score 0.5 (Partially Correct):** The solution has the right general approach but contains minor errors, or is missing some non-trivial steps that affect the completeness of the argument. The core idea is correct but the execution has flaws.

**Score 0 (Incorrect):** The solution has fundamental errors in reasoning, arrives at a wrong conclusion, completely fails to address the problem, or is missing critical steps that make the proof invalid.

**Important:** Focus on whether the mathematical reasoning is correct and complete enough to be convincing. Do not penalize for stylistic choices or for omitting proofs of well-established facts.

Please analyze the solution and provide your evaluation in the following format:

Here is my evaluation of the solution:
... // Analyze the key steps: Are they mathematically correct? Is the logical flow valid? Are there any errors?

Based on my evaluation, the final score is:
\\boxed{{...}} // 0, 0.5, or 1

---
## Problem
{question}
## Solution
{proof}"""


class DeepSeekAPIRewardWorker(Worker):
    """
    Reward Worker using DeepSeek API (or compatible OpenAI-style API) to score mathematical proofs.

    Uses HTTP requests to remote API - no local GPU required.
    All GPU resources can be dedicated to actor training and inference.
    """

    # API configuration
    MAX_RETRIES = 3
    RETRY_SLEEP_SECONDS = 2
    REQUEST_TIMEOUT = 60  # 1 minute timeout for API request

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.actor_tokenizer = None

        # API configuration
        self.api_url = None
        self.api_key = None
        self.model_name = None
        self.generation_config = {}
        self.proxies = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)

        # Get API configuration from worker config
        self.api_url = self.worker_config.judge_api_url
        self.api_key = self.worker_config.judge_api_key
        self.model_name = self.worker_config.judge_model_name

        if not self.api_url:
            raise ValueError("judge_api_url must be configured for DeepSeekAPIRewardWorker")
        if not self.api_key:
            raise ValueError("judge_api_key must be configured for DeepSeekAPIRewardWorker")
        if not self.model_name:
            raise ValueError("judge_model_name must be configured for DeepSeekAPIRewardWorker")

        # Ensure API URL format
        self.api_url = self.api_url.rstrip("/")
        if not self.api_url.endswith("/v1"):
            self.api_url = self.api_url + "/v1"

        # Get generation config
        if hasattr(self.worker_config, 'generating_args') and self.worker_config.generating_args:
            self.generation_config = self.worker_config.generating_args.to_dict()

        # Setup proxy: first try config, then environment variables
        config_proxy = getattr(self.worker_config, 'judge_api_proxy', None)
        if config_proxy:
            self.proxies = {'http': config_proxy, 'https': config_proxy}
            self.logger.info(f"  Using proxy from config: {config_proxy}")
        else:
            http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
            https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
            if http_proxy or https_proxy:
                self.proxies = {}
                if http_proxy:
                    self.proxies['http'] = http_proxy
                if https_proxy:
                    self.proxies['https'] = https_proxy
                self.logger.info(f"  Using proxies from env: {self.proxies}")

        self.logger.info(f"{self.worker_name} initialized with DeepSeek API mode")
        self.logger.info(f"  API URL: {self.api_url}")
        self.logger.info(f"  Model: {self.model_name}")

    # Internal marker for extraction failure
    _EXTRACTION_FAILED = -999.0

    def _extract_score(self, llm_response: str) -> float:
        """Extract score from verifier's response containing \\boxed{score}."""
        if not llm_response or not llm_response.strip():
            return self._EXTRACTION_FAILED

        VALID_SCORES = [0.0, 0.5, 1.0]

        try:
            content = extract_last_boxed(llm_response)

            if content is None:
                # Fallback: look for "score: X" pattern
                matches = re.findall(r'(?:final\s+)?score[:\s]+(-?[0-9.]+)', llm_response, re.IGNORECASE)
                if matches:
                    score = float(matches[-1])
                    if score in VALID_SCORES:
                        return score
                    return min(max(score, 0.0), 1.0)

                # Fallback: find last occurrence of 0, 0.5, or 1
                candidates = []
                for s in ["0.5", "1", "0"]:
                    idx = llm_response.rfind(s)
                    if idx != -1:
                        candidates.append((idx, s))
                if candidates:
                    candidates.sort()
                    s = candidates[-1][1]
                    return float(s)

                return self._EXTRACTION_FAILED

            score = float(content.strip())
            if score in VALID_SCORES:
                return score

            self.logger.warning(f"Unexpected score value: {score}, clamping to [0, 1]")
            return min(max(score, 0.0), 1.0)

        except ValueError:
            self.logger.warning(f"Could not parse score from boxed content: '{content}'")
            return self._EXTRACTION_FAILED
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return self._EXTRACTION_FAILED

    def _build_verification_prompt(self, question: str, proof: str) -> str:
        """Build the verification prompt."""
        return PROMPT_VERIFICATION.format(question=question, proof=proof)

    def _extract_problem_from_prompt(self, prompt: str) -> str:
        """Extract the pure math problem from the actor's prompt."""
        problem = prompt

        # Remove chat template markers
        if "user\n" in problem:
            problem = problem.split("user\n")[-1].strip()

        # Remove instruction prefix
        marker = "the solution is incomplete."
        if marker in problem:
            idx = problem.find(marker)
            problem = problem[idx + len(marker):].strip()

        return problem

    def _call_api(self, question: str, proof: str) -> Dict[str, Any]:
        """
        Call the DeepSeek API to verify a proof.

        Returns:
            Dict with 'score' and 'verification' fields
        """
        prompt = self._build_verification_prompt(question, proof)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Get generation parameters
        temperature = self.generation_config.get("temperature", 0.6)
        max_tokens = self.generation_config.get("max_new_tokens", 8192)

        body = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": 1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.REQUEST_TIMEOUT,
                    proxies=self.proxies
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"API call failed, status={resp.status_code}, body={resp.text[:500]}"
                    )

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("API response does not contain 'choices' field")

                content = choices[0]["message"]["content"]
                score = self._extract_score(content)

                return {
                    "score": score,
                    "verification": content,
                }

            except Exception as e:
                self.logger.warning(f"API call attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_SLEEP_SECONDS)
                else:
                    return {
                        "score": self._EXTRACTION_FAILED,
                        "verification": f"API call failed after {self.MAX_RETRIES} retries: {e}",
                    }

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        """
        Compute rewards using DeepSeek API for verification.
        """
        import json

        # Decode prompts and responses
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        prompt_ids = data.non_tensor_batch["id"]

        scores = []
        failed_extractions = []

        for i, (prompt_txt, response) in enumerate(zip(prompts_text_list, response_text_list)):
            # Extract solution (remove <think> tags if present)
            solution = extract_solution_after_think(response)

            # Extract pure problem from prompt
            problem = self._extract_problem_from_prompt(prompt_txt)

            # Call API
            result = self._call_api(problem, solution)
            score = result["score"]

            # Handle extraction failure
            if score == self._EXTRACTION_FAILED:
                failed_extractions.append({
                    "prompt_id": prompt_ids[i],
                    "problem": problem[:200] + "..." if len(problem) > 200 else problem,
                })
                score = 0.0  # Treat as incorrect

            scores.append(score)

            # Debug logging
            info = {
                "prompt_id": prompt_ids[i],
                "score": score,
            }
            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

        # Log failed extractions
        if failed_extractions:
            self.logger.warning(
                f"Score extraction failed for {len(failed_extractions)}/{len(scores)} samples (assigned 0.0)"
            )

        # Create output tensors
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        # Compute metrics
        n_samples = len(scores)
        mean_reward = scores_tensor.float().mean().item()

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        # Metrics
        metrics = {
            "reward/mean": mean_reward,
            "reward/score_1_ratio": sum(1 for s in scores if s == 1.0) / n_samples,
            "reward/score_0.5_ratio": sum(1 for s in scores if s == 0.5) / n_samples,
            "reward/score_0_ratio": sum(1 for s in scores if s == 0.0) / n_samples,
            "reward/extraction_failed": len(failed_extractions),
            "reward/extraction_failed_ratio": len(failed_extractions) / n_samples,
        }

        output.meta_info = {"metrics": metrics}

        self.logger.info(
            f"DeepSeek API: mean={mean_reward:.4f}, "
            f"score_1={metrics['reward/score_1_ratio']:.2%}, "
            f"score_0.5={metrics['reward/score_0.5_ratio']:.2%}, "
            f"score_0={metrics['reward/score_0_ratio']:.2%}, "
            f"failed={len(failed_extractions)}"
        )

        return output
