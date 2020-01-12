from model import sentence_match
model=sentence_match.SiameseNetwork(pattern='train')
model.train_model()