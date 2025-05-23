# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from typing import List

from .beam_group import BeamGroup
from .greedy_token_selection_strategy import GreedyTokenSelectionStrategy, GreedyBeam

from ..messages import LlmInferenceExecRequest, InferencePhase

logger = logging.getLogger(__name__)


class MultiGreedyTokenSelectionStrategy(GreedyTokenSelectionStrategy):
    def select_greedy(
        self,
        active_beams: List[GreedyBeam],
        _: List[GreedyBeam],
    ) -> List[GreedyBeam]:
        """Greedily select a token for each active beam.

        Args:
            active_beams (List[GreedyBeam]): Beams that are still active.
            _ (List[GreedyBeam]): Beams that are completed.

        Returns:
            List[GreedyBeam]: Beams with new token selected.
        """
        selections = []
        for beam in active_beams:
            token = beam.sample_logits()
            beam.last_token = token
            selections.append(
                beam,
            )

        return selections

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        """Orchestrate decode loop for `multi_greedy` selection strategy.

        Args:
            exec_req (LlmInferenceExecRequest): Initial inference request, post prefill.
        """
        logger.debug("Starting `multi_greedy` decode loop...")
        config = self.token_selection_strategy_config

        if config.decode_config.top_k is not None:
            logger.debug(
                f"Using `top_k` sampling with `top_k == {config.decode_config.top_k}"
            )

        if config.decode_config.top_p is not None:
            logger.debug(
                f"Using `top_p` sampling with `top_p == {config.decode_config.top_p}"
            )

        exec_req.reset(InferencePhase.DECODE)

        # Copy `exec_req` to `num_beams` total requests
        exec_reqs = self.replicate_inference_exec_requests(
            exec_req, config.decode_config.num_beams - 1
        )

        beams = [
            GreedyBeam(exec_req, decode_config=config.decode_config)
            for exec_req in exec_reqs
        ]
        beam_group = BeamGroup(
            config.eos_token_id,
            config.decode_config.num_beams,
            beams,
            self.select_greedy,
        )

        reservations = beam_group.active_beam_count
        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if exec_req.status_tracker.is_disconnected():
                break
            if not beam_group.active_beams:
                break
            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                release_amount = reservations - active_beam_count
                config.decode_end_callback(
                    rid=exec_req.orig_instance_id, count=release_amount
                )
                reservations = active_beam_count

            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)

            await beam_group.wait()
            beam_group.process_beams()

        config.decode_end_callback(rid=exec_req.orig_instance_id, count=reservations)
        beam_group.clean_up()

        results = [
            beam.exec_req.input_token_ids[exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            results.extend(
                [
                    beam.exec_req.input_token_ids[exec_req.prompt_length :]
                    for beam in beam_group.active_beams
                ]
            )
        config.results_callback(results)
