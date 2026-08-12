import torch as th
import torch.nn as nn

from datetime import datetime

from torch.optim import Adam

from jarl.modules import MLP
from jarl.modules.layer import orthogonal_init
from jarl.modules.operator import Critic

from jarl.store import RolloutBuffer
from jarl.collect import (
    LogProbCapture,
    SelfPlayMatchmaker,
    SelfPlayRunner
)
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    TransformRollout,
    Update
)
from jarl.log.logger import Logger
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.transform import GAE

from carl.gymnasium import CARLTorchVectorEnv

from config import ASEConfig, get_config
from expert_dataset import ExpertStateResetProvider, ExpertTrajectoryDataset
from encoder import (
    CARLDiscriminatorEncoder,
    ExpertDiscriminatorEncoder,
    LatentEncoder
)
from modules import (
    ASEDiscriminator,
    LatentMultiCategoricalPolicy,
    SkillEncoder
)
from capture import LatentCapture, LatentCriticCapture
from loss import (
    ASEPPOLoss,
    ASEReward,
    DiscriminatorLoss,
    DiscriminatorMinibatches,
    SkillEncoderLoss
)


CONFIG: ASEConfig = get_config()
DEVICE: th.device = "cuda:0"


env = CARLTorchVectorEnv(
    n_sim=CONFIG.n_sim,
    n_blue=CONFIG.n_blue,
    n_orange=CONFIG.n_orange,
    seed=CONFIG.seed,
    frameskip=CONFIG.frameskip,
    max_ticks=CONFIG.max_ticks,
    normalize=True
)


latent_capture = LatentCapture(
    latent_dim=CONFIG.latent_dim,
    device=DEVICE,
    seed=CONFIG.seed
)


low_level_policy = LatentMultiCategoricalPolicy(
    foot=LatentEncoder(latent_capture),
    body=MLP(
        dims=[1024, 1024, 512],
        func=nn.ReLU
    ),
    head=MLP(
        dims=[],
        out_init_func=orthogonal_init(0.01)
    ),
    action_codec=env.action_codec
).build(env).to(DEVICE)


low_level_critic = Critic(
    foot=LatentEncoder(latent_capture),
    body=MLP(
        dims=[1024, 1024, 512],
        func=nn.ReLU
    ),
    head=MLP(
        dims=[],
        out_init_func=orthogonal_init(0.01)
    )
).build(env).to(DEVICE)


discriminator = ASEDiscriminator(
    foot=CARLDiscriminatorEncoder(),
    expert_foot=ExpertDiscriminatorEncoder(),
    body=MLP(
        dims=[1024, 512],
        func=nn.ReLU
    ),
    head=MLP(dims=[])
).build(env).to(DEVICE)


skill_encoder = SkillEncoder(
    foot=CARLDiscriminatorEncoder(),
    body=MLP(
        dims=[1024, 512],
        func=nn.ReLU
    ),
    head=MLP(dims=[]),
    latent_dim=CONFIG.latent_dim
).build(env).to(DEVICE)


buffer = RolloutBuffer(
    horizon=CONFIG.rollout,
    num_envs=env.n_envs,
    device=DEVICE,
    copy_on_finish=False
)

matchmaker = SelfPlayMatchmaker(
    num_matches=CONFIG.n_sim,
    team_sizes=(CONFIG.n_blue, CONFIG.n_orange),
    current_fraction=1.0,
    historical_ids=(),
    device=DEVICE,
    seed=CONFIG.seed,
)

runner = SelfPlayRunner(
    env=env,
    policy=low_level_policy,
    buffer=buffer,
    opponent_pool=None,
    matchmaker=matchmaker,
    captures=(
        latent_capture,
        LogProbCapture(),
        LatentCriticCapture(low_level_critic, latent_capture)
    )
)


def build_learner(expert_data: ExpertTrajectoryDataset) -> Algorithm:
    policy_optimizer = Adam(
        low_level_policy.parameters(),
        lr=CONFIG.learning_rate
    )

    critic_optimizer = Adam(
        low_level_critic.parameters(),
        lr=CONFIG.learning_rate
    )

    discriminator_optimizer = Adam(
        discriminator.parameters(),
        lr=CONFIG.learning_rate
    )

    skill_optimizer = Adam(
        skill_encoder.parameters(),
        lr=CONFIG.learning_rate
    )

    reward = TransformRollout(
        ASEReward(
            discriminator,
            skill_encoder,
            beta=CONFIG.beta,
            kappa=CONFIG.kappa
        ),
        GAE(gamma=CONFIG.gamma, lambda_=CONFIG.gae_lambda),
        report_fields=("reward", "imitation_reward", "skill_reward"),
        section="ASE Reward"
    )

    discriminator_update = Update(
        transforms=(),
        sampler=DiscriminatorMinibatches(
            expert_data,
            batch_size=CONFIG.auxiliary_batch,
            epochs=CONFIG.auxiliary_epochs
        ),
        loss=DiscriminatorLoss(
            discriminator,
            gradient_penalty=CONFIG.gradient_penalty
        ),
        optimizer_step=OptimizerStep(
            discriminator,
            discriminator_optimizer,
            max_grad_norm=CONFIG.max_grad_norm
        ),
        section="Discriminator"
    )

    skill_update = Update(
        transforms=(),
        sampler=RolloutMinibatches(
            batch_size=CONFIG.auxiliary_batch,
            epochs=CONFIG.auxiliary_epochs
        ),
        loss=SkillEncoderLoss(skill_encoder, kappa=CONFIG.kappa),
        optimizer_step=OptimizerStep(
            skill_encoder,
            skill_optimizer,
            max_grad_norm=CONFIG.max_grad_norm
        ),
        section="Skill Encoder"
    )

    ppo_update = Update(
        transforms=(),
        sampler=RolloutMinibatches(
            batch_size=CONFIG.minibatch_size,
            epochs=CONFIG.epochs
        ),
        loss=ASEPPOLoss(
            low_level_policy,
            low_level_critic,
            PPOConfig(
                clip=CONFIG.clip,
                entropy_coef=CONFIG.entropy_coef
            ),
            diversity=CONFIG.diversity
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(
                low_level_policy,
                policy_optimizer,
                max_grad_norm=CONFIG.max_grad_norm
            ),
            OptimizerStep(
                low_level_critic,
                critic_optimizer,
                max_grad_norm=CONFIG.max_grad_norm
            )
        ),
        section="PPO"
    )

    return Algorithm(
        reward,
        discriminator_update,
        skill_update,
        ppo_update
    )


def main() -> None:
    expert_data = ExpertTrajectoryDataset(
        CONFIG.expert_dir,
        device=DEVICE
    )

    env.reset_state_provider = ExpertStateResetProvider(
        expert_data,
        device=DEVICE
    )

    learner = build_learner(expert_data)
    run_id = datetime.now().strftime("ase-%Y%m%d-%H%M%S")
    logger = Logger(log_dir=str(CONFIG.tensorboard_dir / run_id))

    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger
    )

    trainer.run(CONFIG.total_timesteps)

if __name__ == "__main__":
    main()
