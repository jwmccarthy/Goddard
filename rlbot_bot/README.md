# Goddard RLBot

The Windows deployment lives at `C:\Users\jack\RLBotGoddard` and uses the latest
Runpod checkpoint copied to `policy_latest.pt`. In RLBot v5, choose **Load from
file** and select `C:\Users\jack\RLBotGoddard\bot.toml`.

Launch from WSL with:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  'C:\Users\jack\RLBotGoddard\run.ps1'
```

The adapter runs deterministic policy inference at 15 Hz, reproduces CARL's
normalized 137-feature 1v1 observation, and expects a 1v1 match.
