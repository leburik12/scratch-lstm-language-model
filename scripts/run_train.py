from src.train import train
from src.utils import load_text

text = load_text("data/pride_and_prejudice.txt")
train(text, epochs=5)
