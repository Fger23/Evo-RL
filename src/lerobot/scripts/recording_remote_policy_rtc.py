# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote policy client used by recording loops.

This adapter talks to ``lerobot.async_inference.policy_server`` but leaves robot
control and dataset writing inside ``recording_loop.py`` so human-in-loop
takeover semantics stay in one place.
"""

import logging
import math
import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker

if TYPE_CHECKING:
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.processor import RobotAction
    from lerobot.robots import Robot
else:
    RobotAction = dict[str, Any]


AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}

RTC_INFERENCE_DELAY_KEY = "__rtc_inference_delay"
RTC_PREV_CHUNK_LEFT_OVER_KEY = "__rtc_prev_chunk_left_over"
RTC_EXECUTION_HORIZON_KEY = "__rtc_execution_horizon"


@dataclass
class RemotePolicyRecordConfig:
    """Configuration for using a remote async policy while recording."""

    enable: bool = False
    server_address: str = "localhost:8080"
    policy_type: str = ""
    pretrained_name_or_path: str = ""
    policy_device: str = "cpu"
    client_device: str = "cpu"
    actions_per_chunk: int = 1
    chunk_size_threshold: float = 0.0
    aggregate_fn_name: str = "latest_only"
    obs_queue_timeout_s: float = 2.0
    rename_map: dict[str, str] = field(default_factory=dict)
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=False,
            execution_horizon=10,
            max_guidance_weight=1.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )
    action_queue_size_to_get_new_actions: int = 30

    def __post_init__(self) -> None:
        if not self.enable:
            return
        if not self.server_address:
            raise ValueError("`remote_policy.server_address` cannot be empty.")
        if not self.policy_type:
            raise ValueError("`remote_policy.policy_type` is required when remote policy is enabled.")
        if not self.pretrained_name_or_path:
            raise ValueError(
                "`remote_policy.pretrained_name_or_path` is required when remote policy is enabled."
            )
        if self.actions_per_chunk <= 0:
            raise ValueError("`remote_policy.actions_per_chunk` must be positive.")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("`remote_policy.chunk_size_threshold` must be between 0 and 1.")
        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"Unknown `remote_policy.aggregate_fn_name={self.aggregate_fn_name}`. "
                f"Available: {sorted(AGGREGATE_FUNCTIONS)}"
            )
        if self.obs_queue_timeout_s < 0:
            raise ValueError("`remote_policy.obs_queue_timeout_s` must be non-negative.")
        if self.action_queue_size_to_get_new_actions < 0:
            raise ValueError("`remote_policy.action_queue_size_to_get_new_actions` must be non-negative.")


class RemotePolicyActionClient:
    """Remote async policy client used while the local record loop owns robot control."""

    def __init__(self, cfg: RemotePolicyRecordConfig, robot: "Robot", fps: int):
        if not cfg.enable:
            raise ValueError("RemotePolicyActionClient requires `cfg.enable=true`.")

        self.cfg = cfg
        self.robot = robot
        self.environment_dt = 1 / fps
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        import grpc

        self.services_pb2 = services_pb2
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.action_queue: list["TimedAction"] = []
        self.rtc_action_queue = ActionQueue(cfg.rtc) if cfg.rtc.enabled else None
        self.latency_tracker = LatencyTracker() if cfg.rtc.enabled else None
        self.action_queue_lock = threading.Lock()
        self.action_available = threading.Condition(self.action_queue_lock)
        self.latest_action_timestep = -1
        self.aggregate_fn = AGGREGATE_FUNCTIONS[cfg.aggregate_fn_name]
        self.shutdown_event = threading.Event()
        self.receiver_thread: threading.Thread | None = None
        self.awaiting_actions = False
        self.reset_generation = 0
        self.pending_request_started_t: float | None = None
        self.pending_action_index_before_inference = 0

    @property
    def policy_id(self) -> str:
        return self.cfg.pretrained_name_or_path or self.cfg.policy_type or "remote_policy"

    def start(self) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features

        self.stub.Ready(self.services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            policy_type=self.cfg.policy_type,
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy_device,
            rename_map=self.cfg.rename_map,
            rtc_config=self.cfg.rtc if self.cfg.rtc.enabled else None,
        )
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        self.shutdown_event.clear()
        self.receiver_thread = threading.Thread(
            target=self._receive_actions_loop,
            name="remote-policy-action-receiver",
            daemon=True,
        )
        self.receiver_thread.start()
        logging.info("Remote policy recording client connected to %s.", self.cfg.server_address)

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.receiver_thread is not None:
            self.receiver_thread.join(timeout=self.cfg.obs_queue_timeout_s + 1.0)
        self.channel.close()

    def reset(self) -> None:
        with self.action_available:
            self.action_queue.clear()
            if self.cfg.rtc.enabled:
                self.rtc_action_queue = ActionQueue(self.cfg.rtc)
                self.latency_tracker = LatencyTracker()
            self.latest_action_timestep = -1
            self.awaiting_actions = False
            self.pending_request_started_t = None
            self.pending_action_index_before_inference = 0
            self.reset_generation += 1
            self.action_available.notify_all()

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            if self.awaiting_actions:
                return False
            if self.cfg.rtc.enabled:
                return self.rtc_action_queue.qsize() <= self.cfg.action_queue_size_to_get_new_actions
            if not self.action_queue:
                return True
            return len(self.action_queue) / self.cfg.actions_per_chunk <= self.cfg.chunk_size_threshold

    def _build_rtc_metadata(self) -> dict[str, Any]:
        if not self.cfg.rtc.enabled:
            return {}

        prev_actions = self.rtc_action_queue.get_left_over()
        inference_latency = 0.0 if self.latency_tracker is None else self.latency_tracker.max()
        inference_delay = math.ceil(inference_latency / self.environment_dt)
        return {
            RTC_INFERENCE_DELAY_KEY: inference_delay,
            RTC_PREV_CHUNK_LEFT_OVER_KEY: prev_actions,
            RTC_EXECUTION_HORIZON_KEY: self.cfg.rtc.execution_horizon,
        }

    def _send_observation(
        self,
        observation: dict,
        timestep: int,
        task: str | None,
        must_go: bool,
        extra_observation: dict[str, Any] | None = None,
    ) -> None:
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        raw_observation = dict(observation)
        if task is not None:
            raw_observation["task"] = task
        if extra_observation is not None:
            raw_observation.update(extra_observation)

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(timestep, 0),
            observation=raw_observation,
            must_go=must_go,
        )
        observation_bytes = pickle.dumps(timed_observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            self.services_pb2.Observation,
            log_prefix="[RECORD_REMOTE_POLICY] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator)

    def _receive_actions_loop(self) -> None:
        while not self.shutdown_event.is_set():
            with self.action_queue_lock:
                request_generation = self.reset_generation

            try:
                actions_chunk = self.stub.GetActions(self.services_pb2.Empty())
            except Exception:
                if not self.shutdown_event.is_set():
                    logging.exception("Remote policy action receiver failed.")
                    self._mark_request_finished(request_generation)
                continue

            if not actions_chunk.data:
                self._mark_request_finished(request_generation)
                continue

            timed_actions = pickle.loads(actions_chunk.data)  # nosec
            with self.action_available:
                if request_generation != self.reset_generation:
                    continue
                if self.cfg.rtc.enabled:
                    self._merge_rtc_actions_locked(timed_actions)
                else:
                    self._merge_actions_locked(timed_actions)
                self.awaiting_actions = False
                self.action_available.notify_all()

    def _mark_request_finished(self, request_generation: int) -> None:
        with self.action_available:
            if request_generation == self.reset_generation:
                self.awaiting_actions = False
                self.pending_request_started_t = None
                self.action_available.notify_all()

    def _merge_actions(self, incoming_actions: list["TimedAction"]) -> None:
        with self.action_available:
            self._merge_actions_locked(incoming_actions)
            self.action_available.notify_all()

    def _merge_actions_locked(self, incoming_actions: list["TimedAction"]) -> None:
        current_actions = {action.get_timestep(): action for action in self.action_queue}
        merged = {}

        for action in self.action_queue:
            if action.get_timestep() > self.latest_action_timestep:
                merged[action.get_timestep()] = action

        for action in incoming_actions:
            timestep = action.get_timestep()
            if timestep <= self.latest_action_timestep:
                continue
            if timestep in current_actions:
                from lerobot.async_inference.helpers import TimedAction

                action = TimedAction(
                    timestamp=action.get_timestamp(),
                    timestep=timestep,
                    action=self.aggregate_fn(current_actions[timestep].get_action(), action.get_action()),
                )
            merged[timestep] = action

        self.action_queue = [merged[timestep] for timestep in sorted(merged)]

    def _merge_rtc_actions_locked(self, incoming_actions: list["TimedAction"]) -> None:
        if not incoming_actions:
            return

        processed_actions = torch.stack([action.get_action().detach().cpu() for action in incoming_actions])
        original_actions = torch.stack(
            [
                (
                    action.get_original_action().detach().cpu()
                    if action.get_original_action() is not None
                    else action.get_action().detach().cpu()
                )
                for action in incoming_actions
            ]
        )

        request_latency = 0.0
        if self.pending_request_started_t is not None:
            request_latency = time.perf_counter() - self.pending_request_started_t
            self.latency_tracker.add(request_latency)

        real_delay = math.ceil(request_latency / self.environment_dt)
        if self.cfg.action_queue_size_to_get_new_actions < self.cfg.rtc.execution_horizon + real_delay:
            logging.warning(
                "`remote_policy.action_queue_size_to_get_new_actions` should be >= "
                "rtc.execution_horizon + observed inference delay (%s + %s).",
                self.cfg.rtc.execution_horizon,
                real_delay,
            )

        self.rtc_action_queue.merge(
            original_actions=original_actions,
            processed_actions=processed_actions,
            real_delay=real_delay,
            action_index_before_inference=self.pending_action_index_before_inference,
        )
        self.pending_request_started_t = None

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> RobotAction:
        if self.cfg.client_device != "cpu" and action_tensor.device.type != self.cfg.client_device:
            action_tensor = action_tensor.to(self.cfg.client_device)
        else:
            action_tensor = action_tensor.cpu()
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def _pop_policy_action(self) -> torch.Tensor:
        if self.cfg.rtc.enabled:
            action = self.rtc_action_queue.get()
            if action is None:
                raise TimeoutError("Remote RTC policy returned no executable actions.")
            return action

        if not self.action_queue:
            raise TimeoutError("Remote policy returned no actions.")
        timed_action = self.action_queue.pop(0)
        self.latest_action_timestep = timed_action.get_timestep()
        return timed_action.get_action()

    def get_action(self, observation: dict, task: str | None, timestep: int) -> RobotAction:
        if self._ready_to_send_observation():
            with self.action_queue_lock:
                if self.cfg.rtc.enabled:
                    must_go = True
                    extra_observation = self._build_rtc_metadata()
                    self.pending_action_index_before_inference = self.rtc_action_queue.get_action_index()
                else:
                    must_go = len(self.action_queue) == 0
                    extra_observation = None
                self.awaiting_actions = True
                self.pending_request_started_t = time.perf_counter()
            try:
                self._send_observation(
                    observation=observation,
                    timestep=timestep,
                    task=task,
                    must_go=must_go,
                    extra_observation=extra_observation,
                )
            except Exception:
                with self.action_queue_lock:
                    self.awaiting_actions = False
                    self.pending_request_started_t = None
                raise

        with self.action_available:
            has_action = (
                not self.rtc_action_queue.empty() if self.cfg.rtc.enabled else bool(self.action_queue)
            )
            if not has_action:
                self.action_available.wait(timeout=self.cfg.obs_queue_timeout_s)

            action_tensor = self._pop_policy_action()
            if self.cfg.rtc.enabled:
                self.latest_action_timestep = timestep
        return self._tensor_to_action(action_tensor)
