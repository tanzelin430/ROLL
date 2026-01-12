"""
Zebra Puzzle Reward Worker for evaluating logic puzzle solutions.

Computes grid-level accuracy by comparing predicted arrangement
with ground truth cell by cell.
"""
import ast
import json
from typing import Optional, Union, Dict, Any

import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.rlvr.rewards.utils import extract_answer_tags


def _parse_answer_dict(answer_str: str) -> Optional[Dict]:
    """
    Parse answer string as a dictionary.

    Tries ast.literal_eval first, then json.loads.

    Args:
        answer_str: String representation of answer dict

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not answer_str:
        return None

    # Try ast.literal_eval first (handles Python dict syntax)
    try:
        result = ast.literal_eval(answer_str)
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError):
        pass

    # Try JSON parsing
    try:
        result = json.loads(answer_str)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


def compute_grid_accuracy(predicted: Dict, ground_truth: Dict) -> float:
    """
    Compute cell-level accuracy between predicted and ground truth grids.

    Both predicted and ground_truth should have:
    - "header": list of column names
    - "rows": list of lists (each row is a list of cell values)

    Args:
        predicted: Predicted arrangement dict
        ground_truth: Ground truth arrangement dict

    Returns:
        Accuracy as float between 0.0 and 1.0
    """
    if not isinstance(predicted, dict) or not isinstance(ground_truth, dict):
        return 0.0

    # Check required keys
    if "rows" not in predicted or "rows" not in ground_truth:
        return 0.0

    gt_rows = ground_truth.get("rows", [])
    pred_rows = predicted.get("rows", [])

    if not gt_rows:
        return 0.0

    num_rows = len(gt_rows)
    num_cols = len(gt_rows[0]) if gt_rows else 0

    if num_rows == 0 or num_cols == 0:
        return 0.0

    # Count correct cells
    correct_cells = 0
    total_cells = num_rows * num_cols

    for i in range(num_rows):
        if i >= len(pred_rows):
            continue
        for j in range(num_cols):
            if j >= len(pred_rows[i]):
                continue
            if pred_rows[i][j] == gt_rows[i][j]:
                correct_cells += 1

    return correct_cells / total_cells


class ZebraPuzzleRewardWorker(Worker):
    """
    Reward worker for Zebra Puzzle (logic puzzle) evaluation.

    Extracts predicted arrangement from <answer>...</answer> tags,
    parses as dict, and computes grid-level accuracy.

    Returns continuous reward between 0.0 and 1.0.
    """

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
    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards for Zebra Puzzle solutions.

        Expects:
        - data.batch["responses"]: Tokenized model responses
        - data.non_tensor_batch["ground_truth"]: Ground truth dicts (as JSON strings or dicts)

        Returns:
        - DataProto with scores (continuous 0-1 accuracy)
        """
        response_text_list = self.tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )

        prompts = self.tokenizer.batch_decode(
            data.batch["prompts"], skip_special_tokens=False
        )
        ground_truths = data.non_tensor_batch["ground_truth"]
        tags = data.non_tensor_batch.get("tag", ["zebra_puzzle"] * len(response_text_list))

        scores = []

        for response, gt_raw, prompt, tag in zip(
            response_text_list, ground_truths, prompts, tags
        ):
            # Clean response
            response_clean = (
                response.replace("<|endoftext|>", "")
                .replace("<pad>", "")
                .replace("<|im_end|>", "")
            )

            # Extract answer from <answer>...</answer> tags
            answer_str = extract_answer_tags(response_clean)

            # Parse predicted answer
            predicted = _parse_answer_dict(answer_str) if answer_str else None

            # Parse ground truth (might be string or dict)
            if isinstance(gt_raw, str):
                ground_truth = _parse_answer_dict(gt_raw)
            else:
                ground_truth = gt_raw

            # Compute accuracy
            if predicted is None or ground_truth is None:
                accuracy = 0.0
            else:
                try:
                    accuracy = compute_grid_accuracy(predicted, ground_truth)
                except Exception as e:
                    self.logger.warning(f"Error computing grid accuracy: {e}")
                    accuracy = 0.0

            scores.append(accuracy)

            # Log debug info
            log_data = {
                "tag": tag,
                "accuracy": accuracy,
                "answer_extracted": answer_str is not None,
                "predicted_parsed": predicted is not None,
                "answer_str": answer_str,
                "response": response_clean,
            }
            self.logger.debug(json.dumps(log_data, ensure_ascii=False))

        # Build output tensors
        token_level_rewards = torch.zeros_like(
            data.batch["responses"], dtype=torch.float16
        )
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        response_level_rewards = scores_tensor.clone()

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        return output
