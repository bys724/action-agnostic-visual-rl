"""
Action Decoder

Behavior Embedding으로부터 Robot Action을 예측하는 디코더
Robot-specific하며, Behavior Encoder와 분리되어 학습 가능

학습 전략:
1. Pretraining: Behavior Encoder만 학습 (human video)
2. Finetuning: Action Decoder만 학습 (robot demos) - 10x faster

References:
- 논문 - Action-Agnostic Visual Behavior Representation.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class ActionDecoderConfig:
    """Action Decoder 설정"""
    # Input
    behavior_dim: int = 768  # Behavior embedding dimension

    # Architecture
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1

    # Output
    action_dim: int = 7  # Robot action dimension (e.g., 6-DoF + gripper)
    action_horizon: int = 1  # Number of future actions to predict

    # Action head type
    head_type: Literal["mlp", "diffusion", "gmm"] = "mlp"

    # Diffusion specific (if head_type == "diffusion")
    diffusion_steps: int = 100
    diffusion_beta_schedule: str = "cosine"

    # GMM specific (if head_type == "gmm")
    num_modes: int = 5


class MLPActionHead(nn.Module):
    """
    Simple MLP Action Head

    직접적인 action regression
    """

    def __init__(self, config: ActionDecoderConfig):
        super().__init__()

        self.config = config

        layers = []
        in_dim = config.behavior_dim

        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim

        # Output layer
        output_dim = config.action_dim * config.action_horizon
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        behavior_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            behavior_embedding: [B, behavior_dim]

        Returns:
            actions: [B, action_horizon, action_dim]
        """
        B = behavior_embedding.shape[0]
        out = self.mlp(behavior_embedding)  # [B, action_horizon * action_dim]
        actions = out.view(B, self.config.action_horizon, self.config.action_dim)
        return actions


class GMMActionHead(nn.Module):
    """
    Gaussian Mixture Model Action Head

    Multi-modal action distribution 모델링
    """

    def __init__(self, config: ActionDecoderConfig):
        super().__init__()

        self.config = config
        self.num_modes = config.num_modes

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(config.behavior_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        output_dim = config.action_dim * config.action_horizon

        # Mode weights (logits)
        self.mode_weights = nn.Linear(config.hidden_dim, config.num_modes)

        # Mode means
        self.mode_means = nn.Linear(
            config.hidden_dim, config.num_modes * output_dim
        )

        # Mode log-variances
        self.mode_logvars = nn.Linear(
            config.hidden_dim, config.num_modes * output_dim
        )

    def forward(
        self,
        behavior_embedding: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            behavior_embedding: [B, behavior_dim]
            sample: if True, sample from GMM; else return mode with highest weight

        Returns:
            dict with:
                - "actions": [B, action_horizon, action_dim]
                - "weights": [B, num_modes]
                - "means": [B, num_modes, action_horizon, action_dim]
                - "logvars": [B, num_modes, action_horizon, action_dim]
        """
        B = behavior_embedding.shape[0]
        H = self.config.action_horizon
        A = self.config.action_dim

        # Backbone
        h = self.backbone(behavior_embedding)  # [B, hidden_dim]

        # Mode parameters
        weights = F.softmax(self.mode_weights(h), dim=-1)  # [B, num_modes]
        means = self.mode_means(h).view(B, self.num_modes, H, A)
        logvars = self.mode_logvars(h).view(B, self.num_modes, H, A)

        # Clamp logvars for stability
        logvars = torch.clamp(logvars, -10, 2)

        if sample:
            # Sample mode
            mode_idx = torch.multinomial(weights, 1).squeeze(-1)  # [B]

            # Get selected mode parameters
            batch_idx = torch.arange(B, device=behavior_embedding.device)
            selected_mean = means[batch_idx, mode_idx]  # [B, H, A]
            selected_logvar = logvars[batch_idx, mode_idx]  # [B, H, A]

            # Sample from Gaussian
            std = torch.exp(0.5 * selected_logvar)
            eps = torch.randn_like(std)
            actions = selected_mean + std * eps
        else:
            # Take mode with highest weight
            mode_idx = weights.argmax(dim=-1)  # [B]
            batch_idx = torch.arange(B, device=behavior_embedding.device)
            actions = means[batch_idx, mode_idx]  # [B, H, A]

        return {
            "actions": actions,
            "weights": weights,
            "means": means,
            "logvars": logvars,
        }

    def log_prob(
        self,
        behavior_embedding: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        GMM log probability 계산 (학습용)

        Args:
            behavior_embedding: [B, behavior_dim]
            target_actions: [B, action_horizon, action_dim]

        Returns:
            log_prob: [B]
        """
        B = behavior_embedding.shape[0]
        H = self.config.action_horizon
        A = self.config.action_dim

        # Get GMM parameters
        output = self.forward(behavior_embedding, sample=False)
        weights = output["weights"]  # [B, num_modes]
        means = output["means"]  # [B, num_modes, H, A]
        logvars = output["logvars"]  # [B, num_modes, H, A]

        # Expand target for broadcasting
        target = target_actions.unsqueeze(1)  # [B, 1, H, A]

        # Gaussian log probability for each mode
        var = torch.exp(logvars)
        log_prob_components = -0.5 * (
            logvars + (target - means) ** 2 / var + torch.log(torch.tensor(2 * 3.14159))
        )  # [B, num_modes, H, A]

        # Sum over action dimensions
        log_prob_modes = log_prob_components.sum(dim=(-1, -2))  # [B, num_modes]

        # Log-sum-exp over modes (with weights)
        log_prob = torch.logsumexp(
            torch.log(weights + 1e-8) + log_prob_modes, dim=-1
        )  # [B]

        return log_prob


class ActionDecoder(nn.Module):
    """
    Action Decoder

    Behavior embedding을 받아 robot action을 예측

    Embodiment-specific: 각 로봇에 맞게 따로 학습
    """

    def __init__(self, config: Optional[ActionDecoderConfig] = None):
        super().__init__()

        self.config = config or ActionDecoderConfig()

        # Action head 선택
        if self.config.head_type == "mlp":
            self.action_head = MLPActionHead(self.config)
        elif self.config.head_type == "gmm":
            self.action_head = GMMActionHead(self.config)
        elif self.config.head_type == "diffusion":
            raise NotImplementedError("Diffusion head not yet implemented")
        else:
            raise ValueError(f"Unknown head type: {self.config.head_type}")

        # Loss function
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        behavior_embedding: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            behavior_embedding: [B, behavior_dim]

        Returns:
            dict with "actions" and other head-specific outputs
        """
        if self.config.head_type == "mlp":
            actions = self.action_head(behavior_embedding)
            return {"actions": actions}
        else:
            return self.action_head(behavior_embedding, **kwargs)

    def compute_loss(
        self,
        behavior_embedding: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Action prediction loss 계산

        Args:
            behavior_embedding: [B, behavior_dim]
            target_actions: [B, action_horizon, action_dim]
            mask: [B, action_horizon] valid timestep mask

        Returns:
            loss: scalar
        """
        if self.config.head_type == "gmm":
            # GMM negative log likelihood
            log_prob = self.action_head.log_prob(behavior_embedding, target_actions)
            loss = -log_prob.mean()
        else:
            # MSE loss
            output = self.forward(behavior_embedding)
            pred_actions = output["actions"]
            loss = self.mse_loss(pred_actions, target_actions)

            if mask is not None:
                mask = mask.unsqueeze(-1)  # [B, H, 1]
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()

        return loss

    def predict(
        self,
        behavior_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Action 예측 (inference용)

        Args:
            behavior_embedding: [B, behavior_dim]

        Returns:
            actions: [B, action_dim] (first timestep only)
        """
        output = self.forward(behavior_embedding)
        actions = output["actions"]

        # Return only first timestep
        return actions[:, 0, :]


class FullPipeline(nn.Module):
    """
    Full Action-Agnostic Visual Policy Pipeline

    BehaviorEncoder + ActionDecoder 통합
    """

    def __init__(
        self,
        behavior_encoder: nn.Module,
        action_decoder: ActionDecoder,
        text_encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.behavior_encoder = behavior_encoder
        self.action_decoder = action_decoder
        self.text_encoder = text_encoder

        # Freeze encoder during decoder training
        if freeze_encoder:
            for param in self.behavior_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        task_text: Optional[str | list[str]] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass

        Args:
            img_prev: [B, 3, H, W]
            img_curr: [B, 3, H, W]
            task_text: task description string(s)
            task_embedding: [B, M, D] precomputed task embedding

        Returns:
            dict with "actions", "behavior_embedding", etc.
        """
        # Get task embedding
        if task_embedding is None and task_text is not None and self.text_encoder is not None:
            task_embedding = self.text_encoder(task_text, device=img_prev.device)

        # Encode behavior
        encoder_output = self.behavior_encoder(img_prev, img_curr, task_embedding)
        behavior_embedding = encoder_output["behavior_embedding"]

        # Decode action
        decoder_output = self.action_decoder(behavior_embedding)

        return {
            **decoder_output,
            "behavior_embedding": behavior_embedding,
        }

    def compute_loss(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        target_actions: torch.Tensor,
        task_text: Optional[str | list[str]] = None,
        task_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Loss 계산

        Args:
            img_prev, img_curr: image pair
            target_actions: [B, action_horizon, action_dim]
            task_text or task_embedding: task description
            mask: valid timestep mask

        Returns:
            loss: scalar
        """
        # Forward pass
        output = self.forward(img_prev, img_curr, task_text, task_embedding)

        # Action loss
        loss = self.action_decoder.compute_loss(
            output["behavior_embedding"],
            target_actions,
            mask,
        )

        return loss

    @torch.no_grad()
    def predict_action(
        self,
        img_prev: torch.Tensor,
        img_curr: torch.Tensor,
        task_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        단일 action 예측 (inference용)

        Args:
            img_prev, img_curr: [1, 3, H, W] or [3, H, W]
            task_text: task description

        Returns:
            action: [action_dim]
        """
        # Add batch dimension if needed
        if img_prev.dim() == 3:
            img_prev = img_prev.unsqueeze(0)
            img_curr = img_curr.unsqueeze(0)

        output = self.forward(img_prev, img_curr, task_text)
        action = output["actions"][0, 0]  # First batch, first timestep

        return action
