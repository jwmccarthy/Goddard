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
last two CARL observation fields.
