"""
PSRO Verifier Reward Worker for mathematical proof evaluation.

This worker uses an ensemble of LoRA verifiers (population) to score mathematical proofs.
Each verifier outputs scores in \\boxed{score} format where score is 0, 0.5, or 1.

Ensemble Strategy: Majority + fallback=0
- If 2/3 or 3/3 verifiers agree → use the consensus score
- If all 3 disagree (0, 0.5, 1) → use 0 (conservative fallback)

This achieves 76.9% accuracy on the test set (vs 74-75% for single verifier).

Architecture: Uses vLLM HTTP server with multi-LoRA support.
- Pipeline starts vLLM server with all LoRA adapters loaded
- Worker sends HTTP requests, specifying which LoRA to use via "model" field
- For each proof, sends 3 requests (one per LoRA), then majority voting
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

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


# Verifier prompt template (same as SFT training data)
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


class PSROVerifierRewardWorker(Worker):
    """
    PSRO Reward Worker using an ensemble of LoRA verifiers to score mathematical proofs.

    Uses vLLM HTTP server with multi-LoRA support.
    For each proof, queries all LoRA verifiers and applies majority voting.
    Ensemble strategy: Majority vote + fallback=0 for no consensus.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.actor_tokenizer = None

        # vLLM server config
        self.vllm_server_url = None
        self.lora_names: List[str] = []  # List of LoRA verifier names (e.g., ["V1", "V2", "V3"])
        self.generation_config = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)

        # Initialize tokenizer for verifier
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        # Get LoRA verifier names from config
        if hasattr(self.worker_config, 'lora_verifiers') and self.worker_config.lora_verifiers:
            self.lora_names = list(self.worker_config.lora_verifiers.keys())
        else:
            raise ValueError("lora_verifiers must be configured for PSROVerifierRewardWorker")

        if len(self.lora_names) < 2:
            raise ValueError(f"At least 2 LoRA verifiers required, got {len(self.lora_names)}")

        # Get vLLM server URL from pipeline_config (set by _maybe_start_vllm_server)
        for key, reward_config in pipeline_config.rewards.items():
            if getattr(reward_config, 'lora_verifiers', None):
                url = getattr(reward_config, 'vllm_server_url', None)
                if url:
                    self.vllm_server_url = url
                    break

        if not self.vllm_server_url:
            raise ValueError("vllm_server_url must be set for PSROVerifierRewardWorker. "
                           "Ensure judge_model_type='vllm_server' and vllm_server_gpu are configured.")

        # Get generation config
        if hasattr(self.worker_config, 'generating_args'):
            self.generation_config = self.worker_config.generating_args.to_dict()

        self.logger.info(f"{self.worker_name} initialized with PSRO vLLM server mode")
        self.logger.info(f"  Server URL: {self.vllm_server_url}")
        self.logger.info(f"  LoRA verifiers: {self.lora_names}")

    # Internal marker for extraction failure
    _EXTRACTION_FAILED = -999.0

    def _extract_score(self, llm_response: str) -> float:
        """Extract score from verifier's response containing \\boxed{score}."""
        if not llm_response or not llm_response.strip():
            return self._EXTRACTION_FAILED

        score_text = extract_solution_after_think(llm_response)
        VALID_SCORES = [0.0, 0.5, 1.0]

        try:
            content = extract_last_boxed(score_text)

            if content is None:
                matches = re.findall(r'(?:final\s+)?score[:\s]+(-?[0-9.]+)', score_text, re.IGNORECASE)
                if matches:
                    score = float(matches[-1])
                    if score in VALID_SCORES:
                        return score
                    return min(max(score, 0.0), 1.0)
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

    def _format_judge_prompt(self, prompt: str, response: str) -> str:
        """Format the verifier prompt using the same template as SFT training."""
        problem = prompt

        if "user\n" in problem:
            problem = problem.split("user\n")[-1].strip()

        marker = "the solution is incomplete."
        if marker in problem:
            idx = problem.find(marker)
            problem = problem[idx + len(marker):].strip()

        user_content = PROMPT_VERIFICATION.format(question=problem, proof=response)
        messages = [{"role": "user", "content": user_content}]

        # Apply chat template with enable_thinking=False (same as qwen3_no_think)
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return formatted

    def _get_majority_score(self, scores: List[float]) -> Tuple[float, str]:
        """
        Compute ensemble score using Majority + fallback=0 strategy.

        Args:
            scores: List of scores from each verifier

        Returns:
            (final_score, consensus_type)
            - consensus_type: "unanimous", "majority", "no_consensus", "all_failed", "single"
        """
        valid_scores = [s for s in scores if s != self._EXTRACTION_FAILED]

        if len(valid_scores) == 0:
            return 0.0, "all_failed"

        if len(valid_scores) == 1:
            return valid_scores[0], "single"

        vote = Counter(valid_scores)
        most_common_score, count = vote.most_common(1)[0]

        if count == len(valid_scores):
            return most_common_score, "unanimous"
        elif count >= 2:
            return most_common_score, "majority"
        else:
            # No consensus (all different) → fallback to 0
            return 0.0, "no_consensus"

    def _vllm_server_inference(self, prompt: str, lora_name: str) -> str:
        """
        Send a single inference request to vLLM server with specific LoRA.

        Args:
            prompt: The formatted prompt
            lora_name: Name of the LoRA adapter to use (e.g., "V1")

        Returns:
            Generated text response
        """
        max_tokens = self.generation_config.get("max_new_tokens", 4096)
        temperature = self.generation_config.get("temperature", 0.7)
        top_p = self.generation_config.get("top_p", 0.8)
        repetition_penalty = self.generation_config.get("repetition_penalty", 1.1)

        payload = {
            "model": lora_name,  # Specify which LoRA to use
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        try:
            response = requests.post(
                f"{self.vllm_server_url}/v1/completions",
                json=payload,
                timeout=180,  # 3 minutes timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
        except requests.exceptions.Timeout:
            self.logger.warning(f"vLLM server request timeout for {lora_name}")
            return ""
        except Exception as e:
            self.logger.error(f"vLLM server request failed for {lora_name}: {e}")
            return ""

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        """
        Compute rewards using ensemble of LoRA verifiers via vLLM HTTP server.

        Strategy: Majority vote + fallback=0 for no consensus.
        """
        import json

        # Decode prompts and responses
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        prompt_ids = data.non_tensor_batch["id"]

        # Prepare prompts for verification
        formatted_prompts = []
        solution_text_list = []
        for prompt_txt, response in zip(prompts_text_list, response_text_list):
            solution = extract_solution_after_think(response)
            solution_text_list.append(solution)
            formatted = self._format_judge_prompt(prompt_txt, solution)
            formatted_prompts.append(formatted)

        # Query all LoRA verifiers for each prompt
        all_scores: Dict[str, List[float]] = {name: [] for name in self.lora_names}

        for i, formatted_prompt in enumerate(formatted_prompts):
            for lora_name in self.lora_names:
                response = self._vllm_server_inference(formatted_prompt, lora_name)
                score = self._extract_score(response)
                all_scores[lora_name].append(score)

        # Compute ensemble scores
        final_scores = []
        consensus_types = {"unanimous": 0, "majority": 0, "no_consensus": 0, "all_failed": 0, "single": 0}

        for i in range(len(formatted_prompts)):
            scores_i = [all_scores[name][i] for name in self.lora_names]
            final_score, consensus_type = self._get_majority_score(scores_i)
            final_scores.append(final_score)
            consensus_types[consensus_type] += 1

            # Debug logging
            info = {
                "prompt_id": prompt_ids[i],
                "individual_scores": {name: all_scores[name][i] for name in self.lora_names},
                "final_score": final_score,
                "consensus_type": consensus_type,
            }
            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

        # Create output tensors
        scores_tensor = torch.tensor(final_scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        # Compute metrics
        mean_reward = scores_tensor.float().mean().item()
        n_samples = len(final_scores)

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
            "reward/score_1_ratio": sum(1 for s in final_scores if s == 1.0) / n_samples,
            "reward/score_0.5_ratio": sum(1 for s in final_scores if s == 0.5) / n_samples,
            "reward/score_0_ratio": sum(1 for s in final_scores if s == 0.0) / n_samples,
            "reward/unanimous_ratio": consensus_types["unanimous"] / n_samples,
            "reward/majority_ratio": consensus_types["majority"] / n_samples,
            "reward/no_consensus_ratio": consensus_types["no_consensus"] / n_samples,
        }

        # Add per-verifier mean scores
        for name in self.lora_names:
            valid_scores = [s for s in all_scores[name] if s != self._EXTRACTION_FAILED]
            if valid_scores:
                metrics[f"reward/{name}_mean"] = sum(valid_scores) / len(valid_scores)

        output.meta_info = {"metrics": metrics}

        self.logger.info(
            f"PSRO Ensemble: mean={mean_reward:.4f}, "
            f"unanimous={consensus_types['unanimous']}, "
            f"majority={consensus_types['majority']}, "
            f"no_consensus={consensus_types['no_consensus']}"
        )

        return output
