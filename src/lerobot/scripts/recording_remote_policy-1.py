# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from lerobot.processor import RobotAction

# from .configs import RobotClientConfig
# from .constants import SUPPORTED_ROBOTS
# from .helpers import (
#     Action,
#     FPSTracker,
#     Observation,
#     RawObservation,
#     RemotePolicyConfig,
#     TimedAction,
#     TimedObservation,
#     get_logger,
#     map_robot_keys_to_lerobot_features,
#     visualize_action_queue_size,
# )


import torch


from lerobot.async_inference.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]

@dataclass
class RemotePolicyRecordConfig:
    """Configuration for using a remote async policy while recording."""

    enable: bool = False
    server_address: str = "localhost:8080"
    policy_type: str = ""
    pretrained_name_or_path: str = ""
    policy_device: str = "cpu"
    client_device: str = "cpu"
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    obs_queue_timeout_s: float = 2.0
    rename_map: dict[str, str] = field(default_factory=dict)

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

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)


class RemotePolicyActionClient:
    """Small synchronous client for remote policy actions during recording."""

    def __init__(self, cfg: RemotePolicyRecordConfig, robot: "Robot", fps: int):
        if not cfg.enable:
            raise ValueError("RemotePolicyActionClient requires `cfg.enable=true`.")
        
        self.config = cfg
        self.robot = robot
        # self.robot.connect()
        self.environment_dt = 1 / fps

        # lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        # self.server_address = config.server_address

        # self.policy_config = RemotePolicyConfig(
        #     config.policy_type,
        #     config.pretrained_name_or_path,
        #     lerobot_features,
        #     config.actions_per_chunk,
        #     config.policy_device,
        # )
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        logging.info(f"Initializing client to connect to server at {cfg.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self.latest_performed_action = None

        self._chunk_size_threshold = cfg.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=fps)

        logging.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            logging.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            # policy_config_bytes = pickle.dumps(self.policy_config)
            # policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            policy_config = RemotePolicyConfig(
                policy_type=self.config.policy_type,
                pretrained_name_or_path=self.config.pretrained_name_or_path,
                lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
                actions_per_chunk=self.config.actions_per_chunk,
                device=self.config.policy_device,
                rename_map=self.config.rename_map,
            )

            # logging.info("Sending policy instructions to policy server")
            # logging.debug(
            #     f"Policy type: {self.policy_config.policy_type} | "
            #     f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
            #     f"Device: {self.policy_config.device}"
            # )

            # self.stub.SendPolicyInstructions(policy_setup)
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
            logging.info("Remote policy recording client connected to %s.", self.config.server_address)

            self.shutdown_event.clear()


            # Create and start action receiver thread
            action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
            # Start action receiver thread
            action_receiver_thread.start()


            return True

        except grpc.RpcError as e:
            logging.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        # self.robot.disconnect()
        # logging.debug("Robot disconnected")

        self.channel.close()
        logging.debug("Client stopped, channel closed")

    def reset(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()  # Reset the action queue
            self.action_queue_size = []

        with self.latest_action_lock:
            self.latest_action = -1
            self.action_chunk_size = -1
            self.latest_performed_action = None

        # Sync server dedup / policy internal state when a new episode starts or
        # control returns to the policy after human takeover.
        self.stub.Ready(services_pb2.Empty())

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        logging.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            logging.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            logging.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        logging.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        logging.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                logging.info(f"len(actions_chunk.data) = {len(actions_chunk.data)}")
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    logging.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    logging.debug(f"Converted actions to device: {client_device}")
                else:
                    logging.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    logging.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    logging.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    logging.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    logging.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                logging.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        # _performed_action = self.robot.send_action(
        #     self._action_tensor_to_action_dict(timed_action.get_action())
        # )
        _performed_action = self._action_tensor_to_action_dict(timed_action.get_action())
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_performed_action = _performed_action

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            logging.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            logging.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, obs: dict, task: str, verbose: bool = False):
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation = dict(obs)
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # Force inference whenever we send an observation. When chunk_size_threshold > 0
            # we also send while the queue still has actions; those obs used to use
            # must_go=False and could be filtered as "too similar", starving GetActions.
            with self.action_queue_lock:
                observation.must_go = True
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            logging.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                logging.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                logging.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            # return raw_observation

        except Exception as e:
            logging.error(f"Error in observation sender: {e}")

    def get_action(self, observation: dict, task: str, timestep: int, verbose: bool = True) ->  RobotAction:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        logging.info("Control loop thread starting")

        _performed_action = None
        # _captured_observation = None
        send_observation_flag = False

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            elif self.latest_performed_action is not None:
                _performed_action = self.latest_performed_action
            else:
                if self._ready_to_send_observation() and not send_observation_flag:
                    logging.warning("action queue is null or latest_performed_action is None, wait for receive actions.")
                    self.control_loop_observation(observation, task, verbose)
                    send_observation_flag = True
                # Wait for a short duration before checking again to avoid busy waiting
                time.sleep(0.001)
                continue  # wait for actions to be available

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation() and not send_observation_flag:
                self.control_loop_observation(observation, task, verbose)

            logging.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            if _performed_action is not None:
                break  # Exit after one iteration for this function; the outer loop can call it repeatedly

        return _performed_action


# @draccus.wrap()
# def async_client(cfg: RobotClientConfig):
#     logging.info(pformat(asdict(cfg)))

#     if cfg.robot.type not in SUPPORTED_ROBOTS:
#         raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

#     client = RobotClient(cfg)

#     if client.start():
#         client.logger.info("Starting action receiver thread...")

#         # Create and start action receiver thread
#         action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

#         # Start action receiver thread
#         action_receiver_thread.start()

#         try:
#             # The main thread runs the control loop
#             client.control_loop(task=cfg.task)

#         finally:
#             client.stop()
#             action_receiver_thread.join()
#             if cfg.debug_visualize_queue_size:
#                 visualize_action_queue_size(client.action_queue_size)
#             client.logger.info("Client stopped")


# if __name__ == "__main__":
#     async_client()  # run the client
