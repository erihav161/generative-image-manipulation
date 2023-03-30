import argparse
import sys
import time
import datasets
# import img_text_composition_models
import numpy as np
# from tensorboardX import SummaryWriter
# import test_retrieval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.utils.data
import torchvision
from torchvision.utils import save_image
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import gimli

def test(data, model, epoch, args):
    model.eval()
    