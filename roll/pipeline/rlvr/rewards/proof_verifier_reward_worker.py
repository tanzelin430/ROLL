"""
Proof Verifier Reward Worker for mathematical proof evaluation.

This worker uses a trained verifier model to score mathematical proofs.
The verifier outputs scores in \boxed{score} format where score is 0, 0.5, or 1.

Supports four modes:
- inference: Local inference with vLLM/HF strategy (original mode)
- cluster: Use external reward_infer cluster (Ray-based, queued batching)
- vllm_server: Use vLLM HTTP server for true continuous batching (recommended)
- api: External API (not implemented)
"""
import re
from typing import Any, Dict, List, Optional, Union

import ray

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.pipeline.rlvr.rewards.utils import extract_last_boxed


def extract_solution_after_think(response: str) -> str:
    """
    Extract the solution part after </think> tag.

    For reasoning models with enable_thinking=True, the output format is:
    <think>
    [reasoning process - may contain errors, exploration]
    </think>
    [final solution - this is what should be verified]

    Args:
        response: Full model response, may or may not contain <think> blocks

    Returns:
        The solution part (after </think> if present, otherwise full response)
    """
    # Find the last </think> tag (in case of multiple think blocks)
    think_end_tag = "</think>"
    last_think_end = response.rfind(think_end_tag)

    if last_think_end != -1:
        # Extract everything after </think>
        solution = response[last_think_end + len(think_end_tag):].strip()
        # If solution is empty (model didn't output anything after think), use full response
        if not solution:
            return response
        return solution

    # No </think> tag found, return full response
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


class ProofVerifierRewardWorker(Worker):
    """
    Reward Worker using a trained Verifier model to score mathematical proofs.

    The verifier was trained via SFT on oracle-labeled proof verification data.
    Output format: evaluation text followed by \\boxed{score} where score is 0, 0.5, or 1.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # For cluster mode: reference to external inference cluster
        self.infer_cluster: Optional[Any] = None

        # For vllm_server mode: model name for API calls
        self.vllm_model_name: Optional[str] = None

        # Get judge model type from config: "inference", "cluster", "vllm_server", or "api"
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "inference"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None
        self.vllm_server_url = (
            self.worker_config.vllm_server_url if hasattr(self.worker_config, "vllm_server_url") else None
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)

        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            self.logger.debug(f"{self.worker_name} initialized with API model")
        elif self.judge_model_type == "inference":
            self.strategy = create_strategy(worker=self)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            self.logger.debug(f"{self.worker_name} initialized with local inference model")
        elif self.judge_model_type == "cluster":
            # Cluster mode: use external reward_infer cluster
            # Tokenizer from reward_infer config (should match verifier model)
            if pipeline_config.reward_infer is not None:
                self.tokenizer = default_tokenizer_provider(model_args=pipeline_config.reward_infer.model_args)
            else:
                # Fallback to worker config
                self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            self.logger.debug(f"{self.worker_name} initialized with cluster mode (no local GPU)")
        elif self.judge_model_type == "vllm_server":
            # vLLM HTTP server mode: true continuous batching via HTTP API
            # Get URL and model name from pipeline_config (set by pipeline after starting server)
            # Find our reward config in pipeline_config.rewards
            for key, reward_config in pipeline_config.rewards.items():
                if getattr(reward_config, 'judge_model_type', None) == 'vllm_server':
                    url = getattr(reward_config, 'vllm_server_url', None)
                    if url:
                        self.vllm_server_url = url
                        # Get model name (vLLM uses model path as default model name)
                        model_path = getattr(reward_config, 'vllm_server_model_path', None)
                        if model_path is None:
                            model_path = reward_config.model_args.model_name_or_path
                        self.vllm_model_name = model_path
                        break

            if not self.vllm_server_url:
                raise ValueError("vllm_server_url must be set when using judge_model_type='vllm_server'. "
                               "Ensure vllm_server_gpu is configured so pipeline can auto-start the server.")
            # Only need tokenizer for chat template formatting
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            self.logger.debug(f"{self.worker_name} initialized with vLLM server mode: {self.vllm_server_url}, model: {self.vllm_model_name}")
        else:
            raise ValueError(f"Unsupported judge_model_type: {self.judge_model_type}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_infer_cluster(self, infer_cluster):
        """
        Set the external inference cluster for cluster mode.

        Args:
            infer_cluster: The Cluster object for reward inference
        """
        self.infer_cluster = infer_cluster
        self.logger.info(f"{self.worker_name} connected to reward_infer cluster")
        self.logger.info(f"infer_cluster type={type(infer_cluster)}, workers={getattr(infer_cluster, 'workers', 'NO_WORKERS')}")

    # Internal marker for extraction failure (will be replaced with 0.0)
    _EXTRACTION_FAILED = -999.0

    def _extract_score(self, llm_response: str) -> float:
        """
        Extract score from verifier's llm_response containing \\boxed{score}.

        Uses the shared extract_last_boxed() function for robust extraction.
        This handles both display math \\[\\boxed{1}\\] and inline \\boxed{1}.

        For verifier with thinking enabled, the score is after </think> tag.

        Args:
            llm_response: The response from the verifier model (NOT the actor's proof)

        Returns:
            float: Score value (0.0, 0.5, or 1.0). Returns _EXTRACTION_FAILED (-999.0) if extraction fails.
        """
        # IMPORTANT: Only extract from llm_response (verifier output), not from actor's proof
        if not llm_response or not llm_response.strip():
            return self._EXTRACTION_FAILED

        # For verifier with thinking, extract score from after </think>
        # The \boxed{score} should be in the solution part, not in thinking
        score_text = extract_solution_after_think(llm_response)

        # Valid scores
        VALID_SCORES = [0.0, 0.5, 1.0]

        try:
            # Use shared extraction function (handles nested braces correctly)
            content = extract_last_boxed(score_text)

            if content is None:
                # Fallback: try to find "score: X" pattern in score_text (after </think>)
                matches = re.findall(r'(?:final\s+)?score[:\s]+(-?[0-9.]+)', score_text, re.IGNORECASE)
                if matches:
                    score = float(matches[-1])
                    if score in VALID_SCORES:
                        return score
                    return min(max(score, 0.0), 1.0)
                return self._EXTRACTION_FAILED

            # Parse the extracted content as a number
            score = float(content.strip())
            if score in VALID_SCORES:
                return score

            # Clamp unexpected values to [0, 1]
            self.logger.warning(f"Unexpected score value: {score}, clamping to [0, 1]")
            return min(max(score, 0.0), 1.0)

        except ValueError:
            # content was not a valid number
            self.logger.warning(f"Could not parse score from boxed content: '{content}'")
            return self._EXTRACTION_FAILED
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return self._EXTRACTION_FAILED

    def _format_judge_prompt(self, prompt: str, response: str, reference: str = None) -> List[Dict]:
        """
        Format the verifier prompt using the same template as SFT training.

        Args:
            prompt: The actor prompt containing proof_gen instruction + actual problem
            response: The generated proof to evaluate
            reference: Not used for proof verification

        Returns:
            List of message dicts for chat template
        """
        # Extract only the actual math problem from the actor prompt
        # Actor prompt format: [PROOF_GEN_INSTRUCTION]\n\n[ACTUAL_PROBLEM]
        problem = prompt

        # Clean chat template artifacts if present
        if "user\n" in problem:
            problem = problem.split("user\n")[-1].strip()

        # Strip proof generation instruction (ends with "the solution is incomplete.")
        marker = "the solution is incomplete."
        if marker in problem:
            idx = problem.find(marker)
            problem = problem[idx + len(marker):].strip()

        # Format using the exact template from SFT training
        user_content = PROMPT_VERIFICATION.format(question=problem, proof=response)

        messages = [{"role": "user", "content": user_content}]
        return messages

    def _batch_inference(self, all_messages: List[List[Dict]]) -> List[str]:
        """
        Run batch inference using the local vLLM model.

        Args:
            all_messages: List of message lists, each for one sample

        Returns:
            List of generated responses
        """
        import torch
        from tensordict import TensorDict
        from roll.distributed.scheduler.protocol import DataProto
        from roll.platforms import current_platform
        from roll.datasets.chat_template import get_chat_template

        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")

        # Format all messages using chat template
        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)

        texts = [chat_template_func(messages) for messages in all_messages]

        # Tokenize all texts with padding
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.worker_config.strategy_args.strategy_config.get("max_model_len", 8192)
        )
        input_ids = tokenized["input_ids"].to(current_platform.device_type)
        attention_mask = tokenized["attention_mask"].to(current_platform.device_type)

        # Create generation config
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # Create DataProto for batch
        data = DataProto(
            batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
        )
        data = data.to(current_platform.device_type)
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

        # Generate
        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output
            else:
                generate_ids = output.batch["input_ids"]

        # Decode results - extract only the generated part for each sample
        results = []
        for i in range(len(texts)):
            # Find the actual input length (excluding padding)
            input_len = attention_mask[i].sum().item()
            generated = generate_ids[i, int(input_len):]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(result.strip())

        return results

    def _cluster_inference(self, all_messages: List[List[Dict]]) -> List[str]:
        """
        Run inference via external reward_infer cluster.

        Simply calls reward_infer.generate() - vLLM handles continuous batching
        automatically when multiple workers call concurrently.

        Args:
            all_messages: List of message lists, each for one sample

        Returns:
            List of generated responses
        """
        import torch
        from tensordict import TensorDict
        from roll.datasets.chat_template import get_chat_template

        if self.infer_cluster is None:
            raise ValueError("infer_cluster not set. Call set_infer_cluster first.")

        # Format all messages using chat template
        template_name = self.worker_config.data_args.template if hasattr(self.worker_config, 'data_args') else "native"
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        texts = [chat_template_func(messages) for messages in all_messages]

        # Get max_model_len from reward_infer config
        max_model_len = 8192
        if hasattr(self.pipeline_config, 'reward_infer') and self.pipeline_config.reward_infer is not None:
            max_model_len = self.pipeline_config.reward_infer.strategy_args.strategy_config.get("max_model_len", 8192)

        # Get generation config
        if hasattr(self.pipeline_config, 'reward_infer') and self.pipeline_config.reward_infer is not None:
            generation_config = self.pipeline_config.reward_infer.generating_args.to_dict()
        else:
            generation_config = self.worker_config.generating_args.to_dict()

        # Tokenize all texts as a batch
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_model_len
        )

        # Create DataProto
        data = DataProto(
            batch=TensorDict(
                {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]},
                batch_size=len(texts)
            )
        )
        data.meta_info = {"generation_config": generation_config}

        self.logger.debug(f"Calling reward_infer.generate with batch_size={len(texts)}")

        # Call generate - vLLM handles batching internally
        result = ray.get(self.infer_cluster.generate.remote(data=data))

        # Decode generated texts
        results = []
        output_ids = result.batch["input_ids"]
        for i in range(len(texts)):
            input_len = tokenized["attention_mask"][i].sum().item()
            generated_ids = output_ids[i, int(input_len):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(generated_text.strip())

        self.logger.debug(f"Received {len(results)} responses from reward_infer")
        return results

    def _vllm_server_inference(self, all_messages: List[List[Dict]]) -> List[str]:
        """
        Run inference via vLLM HTTP server for true continuous batching.

        Uses simple synchronous HTTP requests. Since report_response() is called
        with 256 concurrency, multiple requests will naturally be sent concurrently
        to the vLLM server, which handles continuous batching internally.

        Args:
            all_messages: List of message lists, each for one sample

        Returns:
            List of generated responses
        """
        import requests
        from roll.datasets.chat_template import get_chat_template

        # Get generation config from worker config
        generation_config = self.worker_config.generating_args.to_dict() if hasattr(self.worker_config, 'generating_args') else {}
        max_tokens = generation_config.get("max_new_tokens", 2048)
        temperature = generation_config.get("temperature", 0.3)
        top_p = generation_config.get("top_p", 0.8)
        repetition_penalty = generation_config.get("repetition_penalty", 1.0)

        # Format messages using chat template
        template_name = self.worker_config.data_args.template if hasattr(self.worker_config, 'data_args') else "native"
        chat_template_func = get_chat_template(template_name, self.tokenizer)

        url = f"{self.vllm_server_url.rstrip('/')}/v1/completions"
        results = []

        for idx, messages in enumerate(all_messages):
            prompt = chat_template_func(messages)

            payload = {
                "model": self.vllm_model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            if self.tokenizer.eos_token:
                payload["stop"] = [self.tokenizer.eos_token]

            try:
                resp = requests.post(url, json=payload, timeout=120)
                if resp.status_code != 200:
                    self.logger.error(f"vLLM server error: {resp.status_code} - {resp.text[:500]}")
                    results.append("")
                else:
                    result = resp.json()
                    generated_text = result["choices"][0]["text"]
                    results.append(generated_text.strip())
            except requests.exceptions.Timeout:
                self.logger.error(f"[VLLM] Request {idx+1} TIMEOUT after 120s")
                results.append("")
            except Exception as e:
                self.logger.error(f"[VLLM] Request failed: {e}")
                results.append("")

        return results

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        """
        Compute rewards for the given data using the verifier model.

        This method is called by the pipeline to score generated proofs.
        Uses batch inference for efficiency.
        """
        import torch
        import json

        # Decode prompts and responses directly from batch
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        prompt_ids = data.non_tensor_batch["id"]

        # Prepare all judge prompts for batch inference
        # Extract solution part (after </think>) for verification
        all_messages = []
        solution_text_list = []
        for prompt_txt, response in zip(prompts_text_list, response_text_list):
            # For reasoning models with thinking, only verify the solution part
            solution = extract_solution_after_think(response)
            solution_text_list.append(solution)
            messages = self._format_judge_prompt(prompt_txt, solution)
            all_messages.append(messages)

        # Run batch inference based on mode
        if self.judge_model_type == "inference":
            llm_responses = self._batch_inference(all_messages)
        elif self.judge_model_type == "cluster":
            llm_responses = self._cluster_inference(all_messages)
        elif self.judge_model_type == "vllm_server":
            llm_responses = self._vllm_server_inference(all_messages)
        else:
            raise NotImplementedError(f"Mode {self.judge_model_type} not implemented for ProofVerifierRewardWorker")

        # Extract scores from verifier responses
        scores = []
        failed_extractions = []
        for i, (prompt_id, prompt_txt, response, solution, llm_response) in enumerate(
            zip(prompt_ids, prompts_text_list, response_text_list, solution_text_list, llm_responses)
        ):
            score = self._extract_score(llm_response)

            # Track extraction failures
            if score == self._EXTRACTION_FAILED:
                failed_extractions.append({
                    "prompt_id": prompt_id,
                    "llm_response_tail": llm_response[-300:] if llm_response else "<empty>",
                })
                score = 0.0  # Replace with 0 for training

            scores.append(score)

            info = {
                "prompt_id": prompt_id,
                "score": score,
                "prompt": prompt_txt,
                "response": response,
                "llm_response": llm_response,
            }
            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

        # Log extraction failures
        if failed_extractions:
            self.logger.warning(
                f"Score extraction failed for {len(failed_extractions)}/{len(scores)} samples (assigned 0.0):"
            )
            for fail in failed_extractions[:3]:  # Show first 3 failures
                self.logger.warning(f"  - prompt_id={fail['prompt_id']}, response_tail: {fail['llm_response_tail'][:100]}...")

        # Create tensors in expected format
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        # Log summary
        mean_reward = scores_tensor.float().mean().item()
        self.logger.debug(f"Computed rewards for {len(scores)} samples, mean reward: {mean_reward:.4f}")

        # Return in the correct format expected by the pipeline
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        metrics = {
            "reward/mean": mean_reward,
            "reward/score_1_ratio": sum(1 for s in scores if s == 1.0) / len(scores),
            "reward/score_0.5_ratio": sum(1 for s in scores if s == 0.5) / len(scores),
            "reward/score_0_ratio": sum(1 for s in scores if s == 0.0) / len(scores),
            "reward/extraction_failed": len(failed_extractions),
            "reward/extraction_failed_ratio": len(failed_extractions) / len(scores) if scores else 0,
        }
        output.meta_info = {"metrics": metrics}

        return output
