source venv/bin/activate

python main.py \
  --agent mpr \
  --steps-per-train 2 \
  --multi-step 10 \
  --learning-rate 0.0001 \
  --learn-start 2000 \
  --replay-frequency 1 \
  --evaluation-interval 5000 \
  --T-max 100000 \
  --memory-capacity 100000 \
  --log-frequency 100 \
  --augment \
#  --render