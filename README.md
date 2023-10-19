# illustration-elo-rating-estimator
Prediction Model and Annotation Tools for Estimating Elo Rationg of Illustration.

# How to Run
```bash
pip install trueskill fire matplotlib
pip install torch torchvision torchaudio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

python estimate.py
```

## Reference and Related Work
### LAION-Aesthetics
Predict the rating which people when they were asked “How much do you like this image on a scale from 1 to 10?”.
[simulacra-aesthetic-captions](https://github.com/JD-P/simulacra-aesthetic-captions/tree/main) dataset is used for training.

* [LAION-AI/laion-datasets: Description and pointers of laion datasets](https://github.com/LAION-AI/laion-datasets/tree/main)
* [LAION-Aesthetics | LAION](https://laion.ai/blog/laion-aesthetics/)
* [LAION-AI/aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures](https://github.com/LAION-AI/aesthetic-predictor/tree/main)