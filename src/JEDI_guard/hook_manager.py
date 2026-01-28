# -*- coding: utf-8 -*-
"""
HookManager (Hook Manager)

This file defines the `HookManager` class, an internal component of `Guard`.

[!] Changes:
- `set_intervention_state` now additionally accepts a `dynamic_betas` tensor.
- `_write_hook` passes `dynamic_betas` to the intervention function.

Core responsibilities:
1. Act as the bridge between the model (`model`) and the JEDI processor
   (`JEDILogitsProcessor`).
2. Provide `attach_read_hook`, registering a PyTorch `register_forward_hook`
   on the target layer to "read" hidden states.
3. Provide `set_intervention_state`, allowing the JEDI processor to
   dynamically request "writing" (intervening) on hidden states in the next
   forward pass.
4. Manage read/write hook handles (`handle`) and correctly remove them on
   `detach`, preventing memory leaks and restoring the model's original behavior.
"""

import torch
from torch.nn import Module
from typing import Callable, Optional, List, Dict, Any
import logging

# Set up a logger
logger = logging.getLogger(__name__)


class HookManager:
    """
    Manage PyTorch hooks to read and write activations during model forward passes.
    """

    def __init__(self, model: Module, layer_id: int, device: str = 'cpu'):
        """
        Initialize the hook manager.

        Args:
            model (Module):
                The Hugging Face model to attach hooks to.
            layer_id (int):
                Index of the target layer (e.g., `model.layers[layer_id]` for Llama).
            device (str):
                Device to run computations on.
        """
        self.model = model
        self.layer_id = layer_id
        self.device = device

        # Try to auto-locate the decoder layer list in the model
        self.layer_module = self._find_target_layer(model, layer_id)
        if self.layer_module is None:
            msg = (
                f"Unable to locate layer {layer_id} in the model. "
                "Please check the model structure and layer_id."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Handles used to remove hooks later
        self.read_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.write_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # State variables
        self.captured_activations: List[torch.Tensor] = []
        self.intervention_function: Optional[Callable] = None
        self.intervention_indices: Optional[torch.Tensor] = None
        self.dynamic_betas: Optional[torch.Tensor] = None  # [!] New: store dynamic beta

    def _find_target_layer(self, model: Module, layer_id: int) -> Optional[Module]:
        """
        Attempt to find the target layer module in the model.
        This works for common architectures like Llama, Mistral, Gemma, etc.
        """
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # For Llama, Mistral, Gemma, Phi-3, etc.
                return model.model.layers[layer_id]
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # For GPT-2, GPT-NeoX
                return model.transformer.h[layer_id]
            elif hasattr(model, 'layers'):
                # Fallback if model.layers exists at the top level
                return model.layers[layer_id]
            else:
                logger.warning("Unknown model architecture. Unable to auto-locate 'layers'.")
                return None
        except IndexError:
            logger.error(
                f"Layer index {layer_id} is out of range. The model has "
                f"{len(model.model.layers)} layers."
            )
            return None
        except Exception as e:
            logger.error(f"Error locating target layer: {e}", exc_info=True)
            return None

    def _read_hook(self, module: Module, args: tuple, output: Any):
        """
        "Read" hook (forward hook).
        Runs *after* the target layer's forward pass to capture its output hidden state.
        """
        hidden_state = None
        if isinstance(output, tuple):
            # Most HF models (Llama, etc.) return tuples (hidden_state, caches, ...)
            hidden_state = output[0]
        else:
            # Some models may return a tensor directly
            hidden_state = output

        if hidden_state is None:
            logger.warning(
                f"JEDI read hook received empty output at layer {self.layer_id}."
            )
            return

        # --- Bug fix: ---
        # Distinguish 3D (prefill) vs. 2D (autoregressive) cases
        final_hidden_state_3d = None

        if hidden_state.dim() == 3:
            # 3D: (batch_size, seq_len, hidden_dim) - prefill stage
            # We only care about the last token in the sequence
            final_hidden_state_3d = hidden_state[:, -1:, :].detach().to(self.device, non_blocking=True)

        elif hidden_state.dim() == 2:
            # 2D: (batch_size, hidden_dim) - autoregressive stage (seq_len=1)
            # It is already the last token; add the 'seq_len' dimension
            final_hidden_state_3d = hidden_state.unsqueeze(1).detach().to(self.device, non_blocking=True)

        else:
            # Unexpected case
            logger.warning(
                "JEDI read hook: unexpected hidden state dimensionality: "
                f"{hidden_state.dim()}. Skipping capture."
            )
            return

        # Store a consistent (batch_size, 1, hidden_dim) tensor
        self.captured_activations.append(final_hidden_state_3d)

        # Dynamically attach the "write" hook if requested
        self._dynamically_attach_write_hook(module)

    def _dynamically_attach_write_hook(self, module: Module):
        """
        Attach a "write" hook (forward hook) if an intervention has been requested.
        The "write" hook runs *after* `forward` to modify the output.
        """
        if self.intervention_function and self.write_hook_handle is None:
            # logger.debug(f"Dynamically attaching intervention hook at layer {self.layer_id}.")
            # [!] Change: from pre_hook to hook
            self.write_hook_handle = module.register_forward_hook(
                self._write_hook
            )

    # [!] Change: updated function signature and internal logic
    def _write_hook(self, module: Module, args: tuple, output: Any) -> Any:
        """
        "Write" hook (forward hook).
        Runs *after* the target layer's forward pass to modify its output `output`
        (i.e., the hidden_state).
        """
        # [!] Check all required state
        if (self.intervention_function is None or
                self.intervention_indices is None or
                self.dynamic_betas is None):
            return output  # [!] If intervention not active, return original output

        # 1. Extract hidden_state from output
        original_hidden_state = None
        is_tuple_output = False

        if isinstance(output, tuple):
            original_hidden_state = output[0]
            is_tuple_output = True
        else:
            original_hidden_state = output

        if original_hidden_state is None:
            logger.warning(
                f"JEDI write hook received empty output at layer {self.layer_id}."
            )
            return output

        # 2. Apply the intervention function
        #    intervention_function is responsible for only modifying the sequences
        #    marked True in self.intervention_indices.
        # [!] Pass dynamic_betas
        modified_hidden_state = self.intervention_function(
            original_hidden_state,
            self.intervention_indices,
            self.dynamic_betas
        )

        # 3. Repack and return the modified hidden_state
        if is_tuple_output:
            # [!] Return modified tuple
            return (modified_hidden_state,) + output[1:]
        else:
            # [!] Return modified tensor
            return modified_hidden_state

    def attach_read_hook(self):
        """
        Register a permanent "read" hook on the target layer.
        """
        if self.read_hook_handle:
            logger.warning('"Read" hook already attached. Removing old hook first.')
            self.read_hook_handle.remove()

        self.read_hook_handle = self.layer_module.register_forward_hook(
            self._read_hook
        )
        # logger.debug(f'"Read" hook attached to layer {self.layer_id}.')

    def set_intervention_state(self, func: Callable, indices: torch.Tensor, dynamic_betas: torch.Tensor):
        """
        Called by JEDILogitsProcessor to request activating intervention on the next step.

        [!] Change: add dynamic_betas parameter.
        """
        self.intervention_function = func
        self.intervention_indices = indices  # (B,) bool tensor
        self.dynamic_betas = dynamic_betas  # [!] (B,) float tensor

    def clear_intervention_state(self):
        """
        Called at the start of `generate` to reset intervention state.
        """
        self.intervention_function = None
        self.intervention_indices = None
        self.dynamic_betas = None  # [!] Clear beta

        # "Write" hooks are attached dynamically; remove them each round
        if self.write_hook_handle:
            # logger.debug(f"Clearing intervention hook at layer {self.layer_id}.")
            self.write_hook_handle.remove()
            self.write_hook_handle = None

    def get_last_captured_activation(self) -> Optional[torch.Tensor]:
        """
        Called by JEDILogitsProcessor to get the most recently captured activation.
        """
        if not self.captured_activations:
            return None

        # In a single forward pass, the read hook can fire multiple times in order:
        # "prefill stage -> autoregressive stage".
        # We only care about the *latest* activation (corresponding to the current
        # logits token), so we pop from the end instead of the queue head. Otherwise
        # we would use the earliest prefix token and shift detection by a prompt length.
        return self.captured_activations.pop()

    def clear_captured_activations(self):
        """
        Called at the start of `generate` to clear cached activations from the prior run.
        """
        self.captured_activations.clear()

    def remove_all_hooks(self):
        """
        Called by `Guard.detach` to clean up all hooks.
        """
        if self.read_hook_handle:
            self.read_hook_handle.remove()
            self.read_hook_handle = None

        self.clear_intervention_state()  # This removes write_hook_handle
        self.clear_captured_activations()
        # logger.debug(f"All hooks removed from layer {self.layer_id}.")
