# Goddard RLBot v5

This bot runs GAIFO checkpoints in Rocket League on Windows. It loads the policy
architecture and frame skip from the checkpoint, including either MLP or GRU
policies and 1v1, 2v2, or 3v3 matches. It constructs CARL's normalized native
observations without requiring the CUDA simulator or JARL on the game machine.

1. Copy this directory to your Windows bot directory (for example,
   `C:\Users\jack\RLBotGoddard`). From PowerShell in that directory, install
   Python 3.11+ dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. The bundled `policy_latest.pt` is loaded automatically by `bot.py`. To
   update it from a newer GAIFO training checkpoint, export the policy from
   the training checkout, then copy the updated `rlbot_bot/policy_latest.pt`
   into the Windows bot directory:

   ```bash
   .venv/bin/python -m rlbot_bot.export_policy checkpoints/gaifo/RUN/gaifo_STEP.pt
   ```

   Alternatively, copy a full GAIFO training checkpoint as `policy_latest.pt`
   or set `GODDARD_CHECKPOINT` to its path. The bot loads both formats; the
   bundled export omits the critic, discriminator, and optimizer states.
3. In RLBot v5, choose **Load from file** and select `bot.toml`. For a local
   1v1 against a built-in opponent, use `rlbot.toml` with `run.ps1`.

Use a Soccar match with the same team size as the checkpoint. Boost-pad
positions are checked against the standard layout. The model chooses actions
deterministically every saved `frameskip` physics ticks; controls are held
between decisions. The RLBot packet supplies the dodge window used by the
last two CARL observation fields after the initial jump. During RLBot's
`Jumping` phase, the adapter reconstructs CARL's fresh flip window and its
short-lived wheel-contact flag from the game frame count.

### Live comparison diagnostics

The bot prints `GODDARD_DIAG` JSON lines to its RLBot console during the first
900 policy decisions (about 30 seconds at frameskip 4). Startup shows the
checkpoint path and SHA-256, policy shape, frame skip, map, and mutators.
Decision lines show frame gaps, inference time, the raw ball/car state, the
normalized CARL observation, and the action sent to Rocket League. Full packet
and observation details are printed at kickoff, roughly once per second, and
when the car's air state or jump/boost/powerslide buttons change; shorter samples
appear between them. Save those console lines and compare them against the
**same checkpoint** in CARL. Set `GODDARD_DIAGNOSTICS=0` to disable the console
output after the comparison.
