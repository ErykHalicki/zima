lerobot-train \
  --dataset.repo_id=ehalicki/zima-rubiks-cube \
  --policy.type=act \
  --output_dir=outputs/train/act_zima_test \
  --job_name=act_zima_test \
  --policy.device=mps \
  --wandb.enable=false \
  --policy.repo_id=ehalicki/zima-rubiks-cube-act
