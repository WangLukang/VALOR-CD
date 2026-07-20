from .joint_distillation import JointSoftHardDistillationLoss
from .strong_cam_prior import (
    StrongCAMMatchingPriorLoss,
    cam_teacher_consistency_loss,
    local_matching_prior,
)

__all__ = [
    "JointSoftHardDistillationLoss",
    "StrongCAMMatchingPriorLoss",
    "cam_teacher_consistency_loss",
    "local_matching_prior",
]
