# -*- coding: utf-8 -*-
"""
[Core] Guard class

This file defines the Guard class, the one-stop entry point for the JEDI defense
system. It encapsulates all online defense logic and implements stages 5 and 6
from \"Method Flow.md\".

[!] Changes:
- `Guard` and `JEDILogitsProcessor` now support dynamic intervention strength
  `beta'` based on the CUSUM score `A_t`.
- `JEDILogitsProcessor` now handles `A_t` computation and state tracking.
- `CusumState` no longer auto-resets.

Core responsibilities:
1. Load offline-prepared defense artifacts via `from_artifacts`.
2. Attach to a Hugging Face model via `attach(model)` using a `with` context.
3. Patch the model's `generate` method during `attach` to intercept generation.
4. Inject a custom `LogitsProcessor` (JEDILogitsProcessor) that, at each step:
   a. Gets the current token hidden state from `HookManager` (read hook).
   b. Calls `Scorer` to compute per-step risk score r_t (stages 4.1, 5.3).
   c. [!] Feeds r_t into `CusumState` to retrieve A_t (stage 5.4).
   d. [!] If `A_t > alpha`, activates intervention and computes dynamic `beta'`
      (stage 5.5).
5. Dynamically register a "write" hook via `HookManager` to perform ActAdd
   injection defined in `interventions.py` (stage 6.1).
6. On exiting the `with` block, automatically `detach`, clean up hooks, and
   restore the model's original `generate` method.
"""

import torch
import logging
from torch.nn import Module
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Callable, Optional, List, Dict, Any
from contextlib import contextmanager

from .components.scorer import Scorer
from .components.cusum import CusumState
from .components.representation import apply_transform
from .hook_manager import HookManager
from .interventions import create_intervention_hook_func
from .utils.config_loader import load_defense_artifacts

# Set up a logger
logger = logging.getLogger(__name__)


class JEDILogitsProcessor(LogitsProcessor):
    """
    Core logic processor for JEDI defense.
    Called at each token generation step in the `generate` loop.

    [!] Change: implements dynamic beta computation and intervention state
    management.
    """

    def __init__(self, guard_instance, batch_size: int, trigger_logs: List[int]):
        """
        Initialize JEDI LogitsProcessor.

        Args:
            guard_instance (Guard):
                Reference to the main Guard instance, used to access scorer, cusum,
                and hook_manager.
            batch_size (int):
                Batch size for the current generation request.
            trigger_logs (List[int]):
                A list of length batch_size (held by the Guard instance),
                initialized to -1. This processor records the *first* triggering
                token index in this list.
        """
        self.guard = guard_instance
        self.batch_size = batch_size
        self.device = self.guard.device
        self.is_batch = batch_size > 1

        # --- Logging ---
        self.trigger_logs = trigger_logs  # Shared list reference
        self.current_step = 0  # Track current token index (starting at 0)

        # Initialize a CUSUM state machine for this generation call
        self.cusum = CusumState(
            mu_hat=self.guard.mu_hat,
            kappa=self.guard.kappa,
            alpha=self.guard.alpha,
            batch_size=batch_size,
            device=self.device
        )

        # --- [!] New: dynamic Beta and state management ---
        # Track which sequences have triggered intervention
        self.intervention_active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # Get CUSUM threshold from Guard
        self.alpha = self.guard.alpha
        # Get base intervention strength from Guard
        self.base_beta = self.guard.base_beta
        # Initialize a tensor to store each sequence's *current* intervention strength
        self.dynamic_betas = torch.full(
            (batch_size,), self.base_beta, device=self.device, dtype=torch.float32
        )
        # --- End new ---

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Execute detection and intervention logic at each generation step.

        Args:
            input_ids (torch.LongTensor): Token sequence generated so far.
            scores (torch.FloatTensor): Raw logits for the current step.

        Returns:
            torch.FloatTensor: Possibly modified (or unmodified) logits.
        """
        # 1. Get the newly captured hidden state from the hook (stage 5.3)
        try:
            hidden_state = self.guard.hook_manager.get_last_captured_activation()
            if hidden_state is None:
                logger.warning(
                    "JEDI: Failed to get hidden state from HookManager. Skipping detection."
                )
                self.current_step += 1  # [!] Ensure step counter increments
                return scores

            if hidden_state.dim() == 2:
                hidden_state = hidden_state.unsqueeze(1)  # (B, D) -> (B, 1, D)

            if hidden_state.shape[0] != scores.shape[0]:
                logger.error(
                    "JEDI: Hidden state batch size (%s) does not match logits batch size (%s).",
                    hidden_state.shape[0],
                    scores.shape[0],
                )
                self.current_step += 1  # [!] Ensure step counter increments
                return scores

        except Exception as e:
            logger.error(f"JEDI: Error fetching hidden state: {e}", exc_info=True)
            self.current_step += 1  # [!] Ensure step counter increments
            return scores

        # 2. Compute risk scores (stage 5.3)
        s_t, r_t = self.guard.scorer.calculate_scores(hidden_state)

        # 3. [!] Update CUSUM state and get A_t (stage 5.4)
        # A_t is (B,) tensor on CPU
        A_t = self.cusum.update(r_t)
        A_t_device = A_t.to(self.device)

        # 4. [!] Check triggers and update intervention state

        # 4a. Determine which sequences *currently* should be triggered
        currently_triggered = A_t_device > self.alpha

        # 4b. Determine which are *newly* triggered
        newly_triggered = currently_triggered & (~self.intervention_active)

        if torch.any(newly_triggered):
            # 4c. Mark newly triggered sequences as "permanently" active
            self.intervention_active |= newly_triggered

            # 4d. Record the *first* triggering step
            for i in newly_triggered.nonzero(as_tuple=True)[0]:
                idx = i.item()
                if self.trigger_logs[idx] == -1:  # Double-check to record only once
                    self.trigger_logs[idx] = self.current_step

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "JEDI: CUSUM first triggered at step %s. Activated sequences: %s",
                    self.current_step,
                    newly_triggered.nonzero(as_tuple=True)[0].tolist(),
                )

        # 5. [!] If *any* sequence (new or old) is active, compute beta' and set hook
        if torch.any(self.intervention_active):
            active_indices = self.intervention_active

            # 5a. Compute dynamic Betas
            # (A_t / alpha) then raised to gamma
            ratios = (A_t_device[active_indices] / self.alpha)
            gamma = 5  # or 1.5
            ratios = ratios.pow(gamma)
            self.dynamic_betas[active_indices] = self.base_beta * ratios

            if logger.isEnabledFor(logging.DEBUG):
                if torch.any(newly_triggered):  # Log only on first trigger
                    active_idxs_list = active_indices.nonzero(as_tuple=True)[0].tolist()
                    betas_list = self.dynamic_betas[active_indices].tolist()
                    logger.debug(
                        "  > Betas: %s",
                        {idx: beta for idx, beta in zip(active_idxs_list, betas_list)},
                    )

            # 5b. Tell HookManager to apply the intervention hook on the *next* forward pass
            self.guard.hook_manager.set_intervention_state(
                self.guard.intervention_func,  # Intervention function
                self.intervention_active,  # (B,) bool, which sequences to intervene
                self.dynamic_betas  # (B,) float, beta values for all sequences
            )

        # 6. Increment generation step counter
        self.current_step += 1

        return scores


class Guard:
    """
    Main class for the JEDI defense system.
    Use via the context manager (`with guard.attach(model): ...`).
    """

    def __init__(
            self,
            layer_id: int,
            theta: float,
            mu_hat: float,
            kappa: float,
            alpha: float,
            base_beta: float,  # [!] New: base beta
            scorer: Scorer,
            intervention_func: Callable,
            device: str = 'cpu'
    ):
        """
        Initialize a Guard instance.

        Note: using `Guard.from_artifacts` is recommended.
        """
        self.layer_id = layer_id
        self.device = device

        # CUSUM parameters
        self.mu_hat = mu_hat
        self.kappa = kappa
        self.alpha = alpha

        # [!] Intervention parameters
        self.base_beta = base_beta

        # Core components
        self.scorer = scorer
        self.intervention_func = intervention_func

        # Runtime state (set during `attach`)
        self.model: Optional[Module] = None
        self.hook_manager: Optional[HookManager] = None
        self.original_generate: Optional[Callable] = None

        # --- New: pass log list between Guard and JEDILogitsProcessor ---
        self.current_batch_trigger_logs: Optional[List[int]] = None
        # --- End new ---

        logger.info("Guard instance initialized. Will run on layer %s.", layer_id)
        logger.info(
            "Defense params: theta=%.4f, mu_hat=%.4f, kappa=%.4f, alpha=%.4f, base_beta=%.2f",
            theta,
            mu_hat,
            kappa,
            alpha,
            base_beta,
        )

    @classmethod
    def from_artifacts(cls, artifact_path: str, device: Optional[str] = None) -> "Guard":
        """
        [Recommended] Load and create a Guard instance from an offline artifacts directory.

        Args:
            artifact_path (str):
                Directory containing all required artifact files.
                (defense_params.yaml, transforms.pt,
                 condition_vectors.pt, intervention_vectors.pt)
            device (str, optional):
                Device to run the defense on ('cuda', 'cpu').
                If None, CUDA will be auto-detected.

        Returns:
            Guard: A configured, ready-to-use Guard instance.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Load all files using the helper function
        try:
            artifacts = load_defense_artifacts(artifact_path, device)
            params = artifacts['defense_params']
        except FileNotFoundError as e:
            logger.error(
                "Failed to load defense artifacts: %s. Ensure path '%s' is correct and contains all required files.",
                e,
                artifact_path,
            )
            raise

        # 2. Extract defense parameters (stage 4 calibration result)
        layer_id = params['best_layer']
        theta = params['theta']
        mu_hat = params['mu_hat']
        kappa = params['kappa']
        alpha = params.get('alpha', params.get('h'))
        if alpha is None:
            raise KeyError("Missing CUSUM alert threshold (alpha) in defense_params.yaml.")

        # 3. Prepare Scorer (stage 4.1)
        transform_cont = artifacts['transforms']['content_window'][layer_id]
        condition_vector = artifacts['condition_vectors'][layer_id]
        scorer = Scorer(
            condition_vector=condition_vector,
            transform=transform_cont,  # [!] Pass content_window transform
            theta=theta,
            device=device
        )

        # 4. Prepare Intervention (stage 6.1)
        transform_early = artifacts['transforms']['early_window'][layer_id]
        intervention_vector = artifacts['intervention_vectors'][layer_id]

        # [!] Load *base* beta from defense_params.yaml
        base_beta = params.get('beta', 1.0)  # Try key 'beta'
        if 'intervention_beta' in params:  # Fallback key
            base_beta = params.get('intervention_beta', base_beta)

        logger.info("Using base intervention strength (base_beta): %s", base_beta)

        # [!] Do not pass beta when creating the intervention function
        intervention_func = create_intervention_hook_func(
            vector=intervention_vector,
            # beta=base_beta, <-- [!] removed
            transform=transform_early,  # Pass early_window transform
            device=device
        )

        # 5. Create and return Guard instance
        return cls(
            layer_id=layer_id,
            theta=theta,
            mu_hat=mu_hat,
            kappa=kappa,
            alpha=alpha,
            base_beta=base_beta,  # [!] Pass base beta
            scorer=scorer,
            intervention_func=intervention_func,
            device=device
        )

    # --- New: methods to set and clear log targets ---
    def set_batch_log_target(self, log_list: List[int]):
        """
        Before `_guarded_generate`, set a list (from run_evaluation.py) for
        JEDILogitsProcessor to record trigger steps.
        """
        self.current_batch_trigger_logs = log_list

    def clear_batch_log_target(self):
        """
        After `_guarded_generate`, clear the log target reference.
        """
        self.current_batch_trigger_logs = None

    # --- End new ---

    @contextmanager
    def attach(self, model: Module):
        """
        Attach the defense system to a model as a context manager.

        Usage:
            with guard.attach(model):
                model.generate(...)

        Args:
            model (Module): The Hugging Face model to protect.
        """
        if self.model is not None:
            logger.warning("Guard is already attached to a model. Detaching old model first.")
            self.detach()

        try:
            self.model = model
            self.hook_manager = HookManager(
                model=model,
                layer_id=self.layer_id,
                device=self.device
            )

            # 1. Save original generate method
            self.original_generate = model.generate
            # 2. Patch generate method
            model.generate = self._guarded_generate

            # 3. Attach read hook to capture hidden states
            #    (intervention hook will be attached dynamically by JEDILogitsProcessor)
            self.hook_manager.attach_read_hook()

            logger.info(
                "Guard attached to model %s (layer: %s).",
                model.config.name_or_path,
                self.layer_id,
            )
            yield self  # Enter 'with' block

        finally:
            self.detach()  # Auto-detach when leaving 'with'

    def detach(self):
        """
        Detach the defense system from the model, clean up hooks, and restore
        original methods.
        """
        if self.model and self.original_generate:
            # Restore original generate method
            self.model.generate = self.original_generate
            logger.info("Restored original 'generate' method on the model.")

        if self.hook_manager:
            # Remove all registered PyTorch hooks
            self.hook_manager.remove_all_hooks()
            logger.info("Removed all model hooks.")

        # Clear state
        self.model = None
        self.hook_manager = None
        self.original_generate = None
        self.clear_batch_log_target()  # Ensure log target is cleared too
        logger.info("Guard detached successfully.")

    def _guarded_generate(self, *args, **kwargs) -> Any:
        """
        Patched `generate` method that injects JEDI defense logic.
        """
        if self.hook_manager is None or self.original_generate is None:
            raise RuntimeError(
                "Guard is not attached to a model. Please use `with guard.attach(model): ...`."
            )

        # 1. Determine batch size
        input_ids = None
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif len(args) > 0:
            input_ids = args[0]

        if input_ids is None:
            if 'inputs' in kwargs:
                input_ids = kwargs['inputs']
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                input_ids = args[0]
            else:
                logger.warning(
                    "JEDI: Unable to determine input_ids in generate call. Assuming batch size 1."
                )
                batch_size = 1
        else:
            batch_size = input_ids.shape[0]

        # 2. Clear hook state from previous round
        self.hook_manager.clear_captured_activations()
        self.hook_manager.clear_intervention_state()

        # 3. Initialize JEDI LogitsProcessor and pass log list
        # --- Change: get log list from Guard instance ---
        trigger_logs = self.current_batch_trigger_logs
        if trigger_logs is None:
            logger.warning(
                "JEDI: Guarded generate called without batch log target (set_batch_log_target). "
                "Trigger steps will not be recorded."
            )
            # Create a temporary dummy list to prevent crashes
            trigger_logs = [-1] * batch_size
        elif len(trigger_logs) != batch_size:
            logger.error(
                "JEDI: Provided log list length (%s) does not match batch size (%s).",
                len(trigger_logs),
                batch_size,
            )
            # Use a dummy list as well
            trigger_logs = [-1] * batch_size

        JEDI_processor = JEDILogitsProcessor(
            guard_instance=self,
            batch_size=batch_size,
            trigger_logs=trigger_logs  # Pass shared list
        )
        # --- End change ---

        # 4. Inject our processor into the `generate` call
        processor_list = kwargs.get('logits_processor')
        if processor_list is None:
            processor_list = LogitsProcessorList()
        elif not isinstance(processor_list, LogitsProcessorList):
            processor_list = LogitsProcessorList([processor_list])  # Ensure list

        processor_list.append(JEDI_processor)
        kwargs['logits_processor'] = processor_list

        # 5. Call the original `generate` method
        return self.original_generate(*args, **kwargs)
