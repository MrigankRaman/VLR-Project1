import random
import torch
import os
import argparse
import json

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--N_data', type=int, default=1000, help='Number of data points')
parser.add_argument('--b_size', type=int, default=100, help='Batch size')
parser.add_argument('--N_test', type=int, default=100, help='Number of test points')
parser.add_argument('--L', type=int, default=3, help='Length of the Addition')
parser.add_argument('--q_e', type=float, default=0.0, help='Error probability')
parser.add_argument('--p', type=float, default=0.0, help='Error probability in digit')
parser.add_argument('--q', type=float, default=0, help='Error probability in carrier')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0, help='Learning rate')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--gpt', type=bool, default=False, help='Use GPT or not')
parser.add_argument('--opt', type=str, default="Adam", help='Use GPT or not')
# Parse the arguments
args = parser.parse_args()

# Assign the values of the variables from the arguments
N_data = args.N_data
b_size = args.b_size
N_test = args.N_test
L = args.L
p = args.p
lr = args.lr
epochs = args.epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Local training
try:
    
    output_dir = os.environ['AMLT_OUTPUT_DIR'] + "/trainer_textbook/"
    args.b_size = args.b_size * 2
    #args.per_device_train_batch_size = args.per_device_train_batch_size * 2
except Exception as e:
    #local training, disable deepspeed
    pass
    #args.deepspeed = None


def generate_string(l, p, q=None):
  # if q is not specified, use p
  if q is None:
    q = p

  # generate A and B as l digit arrays
  A = [random.randint(0, 9) for i in range(l)]
  B = [random.randint(0, 9) for i in range(l)]

  # compute C
  C = []
  carrier = 0
  q_e = random.random()
  for i in range(l-1, -1, -1):
    sum = A[i] + B[i] + carrier
    if sum >= 10:
      carrier = 1
      sum -= 10
    else:
      carrier = 0

    # with probability p, sum is off by one digit
    if random.random() < p and q_e < args.q_e:
      if sum == 0:
        sum = random.choice([1, 9])
      elif sum == 9:
        sum = random.choice([8, 0])
      else:
        sum += random.choice([-1, 1])

    # with probability q, carrier is missing
    if random.random() < q and q_e < args.q_e:
      carrier = 0

    C.append(sum)

   # if carrier is still 1, add it to the front
  if carrier == 1:
    C.append(1)

  # compute D
  D = []
  carrier = 0
  for i in range(l-1, -1, -1):
    sum = A[i] + B[i] + carrier
    if sum >= 10:
      carrier = 1
      sum -= 10
    else:
      carrier = 0
    D.append(sum)

  if carrier == 1:
    D.append(1)

  # convert A, B, C, D to strings
  A = ''.join(map(str, A))
  B = ''.join(map(str, B))
  C = ''.join(map(str, reversed(C)))
  D = ''.join(map(str, reversed(D)))

  return A + ' + ' + B + ' = ' + C, A + ' + ' + B + ' = ' + D


#print(generate_string(4, 0.2))

def generate_batch_string(batch_size = b_size, test = False, testing = False):
    error_s = []
    correct_s = []
    error_s2 = []
    for _ in range(batch_size):
        error_string, correct_string = generate_string(L, p, q = args.q)
        if test:
            if args.gpt:
                ind = error_string.find("=")
                error_string = error_string[0:ind + 1]
                ind = correct_string.find("=")
                correct_string = correct_string[ind + 1:]
            else:
                ind = error_string.find("=")
                er = error_string[0:ind + 1]
                if testing:
                    ind = correct_string.find("=")
                    correct_string = correct_string[ind + 1:]
                else:
                    correct_string = error_string[ind + 1:]
                error_string = er
            
        error_s.append(error_string)
        correct_s.append(correct_string)
    return error_s, correct_s


if args.gpt:
    train_data, _ = generate_batch_string(batch_size = N_data)
   
else:
    train_data, train_label = generate_batch_string(batch_size = N_data, test = True)

test_data, test_label = generate_batch_string(batch_size = N_test, test = True, testing = True)
# Import modules
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertForMaskedLM, AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup


if args.gpt:
    # Load a pre-trained GPT2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    config = model.config
    config.hidden_size = 128
    config.num_attention_heads = 8
    config.num_hidden_layers = 4
    model = BertForMaskedLM(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(data):
    #return tokenizer(data, return_tensors="pt", padding=True).to(device)
    seq_len = 0
    for i in range(len(data)):
        seq_len = max(seq_len, len(data[i]))
    
    data_torch = torch.zeros((len(data), seq_len + 2)).long().to(device)
    for i in range(len(data)):
        for s in range(len(data[i])):
            data_torch[i, s + 1] = ord(data[i][s]) + 100

        data_torch[i, 0] = 1

    ret = {'input_ids': data_torch}
    return ret



def decode(data):
    #return tokenizer.batch_decode(data, skip_special_tokens = True)
    #print(data)
    m = data.shape[0]
    n = data.shape[1]
    data_decode = []
    for i in range(m):
        dd = ""
        for j in range(n):
            try:
                dd += chr(data[i, j] - 100)
            except Exception as e:
                dd += ""
        data_decode.append(dd)

    return data_decode



class StringDataset(torch.utils.data.Dataset):
  # Initialize the dataset with the list of strings
  def __init__(self, strings):
    self.strings = strings
  
  # Return the length of the dataset
  def __len__(self):
    return len(self.strings)
  
  # Return the string at a given index
  def __getitem__(self, index):
    return self.strings[index]

#train_data = tokenize(train_data)
#test_data = tokenize(test_data)

#print(decode(train_data))


# Convert the train_data and test_data tensors into torch datasets and dataloaders
train_dataset = StringDataset(train_data)
if not args.gpt:
    train_label = StringDataset(train_label)
test_dataset = StringDataset(test_data)

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        assert len(data) == len(label) # check that the datasets have the same length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

if not args.gpt:
    # create an instance of the custom Dataset
    train_pairs = PairDataset(train_dataset, train_label)

    # create a dataloader from the custom Dataset
    train_dataloader = DataLoader(train_pairs, batch_size=b_size, shuffle=True)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=b_size)

# Define a loss function, an optimizer, and a scheduler
#loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
if args.opt == "Adam":
  optimizer = AdamW(model.parameters(), lr=lr, betas=(0.95, 0.995), weight_decay = args.wd)
elif args.opt == "SGD":
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay = args.wd)
else:
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay = args.wd)
#write SGD optimizer with momentum
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay = args.wd)
total_steps = len(train_dataloader) * epochs # epochs is the number of training epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# define the evaluation function
def evaluate(model = model, test_dataloader = test_dataloader):
  # set the model to evaluation mode
  model.eval()
  # initialize an empty list to store the generated texts
  generated_texts = []
  # loop through the batches in the test dataloader
  for batch_input in test_dataloader:
    # move the batch to the device (cpu or gpu)
    #print(batch_input[0].size())
    # generate outputs from the model with no_grad to save memory
    with torch.no_grad():
      # use the model.generate method with some parameters
      # max_length: the maximum length of the generated sequence
      # eos_token_id: the id of the end of sentence token, which is 0 for gpt2
      # num_return_sequences: the number of generated sequences to return, which is equal to the batch size
      #print(tokenize(batch_input)['input_ids'].shape)
      if args.gpt: 
        outputs = model.generate(tokenize(batch_input)['input_ids'], max_length=100, eos_token_id=tokenizer.eos_token_id, pad_token_id = tokenizer.eos_token_id, do_sample=False)
      #print(outputs.tolist())
      else:
          outputs = model(tokenize(batch_input)['input_ids']).logits.argmax(axis=-1)
      #print(outputs.shape)
    # decode the outputs to texts and append them to the list
    #for output in outputs:
    generated_texts += decode(outputs)
  # return the list of generated texts
  return generated_texts


def fake_addition(input_string):
  # split the input string by the plus sign and the equal sign
  parts = input_string.split("+")
  A = parts[0].strip() # remove any whitespace from A
  B = parts[1].split("=")[0].strip() # remove any whitespace and the equal sign from B

  # initialize the result as an empty string
  result = ""

  # loop through the digits of A and B from right to left
  i = len(A) - 1
  j = len(B) - 1
  while i >= 0 or j >= 0:
    # get the current digits of A and B, or 0 if they are out of bounds
    a = int(A[i]) if i >= 0 else 0
    b = int(B[j]) if j >= 0 else 0

    # add the digits without any carrier, and append the result to the left of the result string
    result = str((a + b) % 10) + result

    # update the indices
    i -= 1
    j -= 1
  
  # return the result as an int
  return int(result)


def check_accuracy(generate_texts):
    correct = 0
    total = 0
    correct2 = 0
    for g in range(len(generate_texts)):
        
        text = generate_texts[g]
        ind = text.find("=")

        try:
            g_fake_val = fake_addition(test_data[g])
            g_val = int(text[ind + 1:].strip().split(" ")[0])
            if g_val == int(test_label[g]):
                correct += 1
            elif g_val == g_fake_val:
                correct2 += 1
            else:
                pass
        except Exception as e:
            pass
        total += 1

    return correct/total, correct2/total


import logging
import sys

# get the logger object with the module name
logger = logging.getLogger(__name__)

# setup logging with the desired format, datefmt, and handlers
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO # set the logging level to INFO
)


import torch.nn.functional as F
# Define a training loop
 # put the model in training mode
print(args.b_size)
print(args.lr)
print(optimizer)
total_losses = []
for epoch in range(epochs):
  total_loss = 0
  if epoch % 10 == 0:
    generate_texts = evaluate()
    acc, acc2 = check_accuracy(generate_texts)
    logger.info(f'Accuracy correct: {acc}')
    logger.info(f'Accuracy fake: {acc2}')
    #print(check_accuracy(generate_texts))
  for batch in train_dataloader:
    model.train()
    if args.gpt:
        # get the input and target tensors from the batch
        input_ids = tokenize(batch) # device is either 'cpu' or 'cuda'
    else:
         train_data, train_label = batch
         input_ids = tokenize(train_data)
         train_ids = tokenize(train_label)['input_ids']
    if args.gpt:
        target_ids = input_ids['input_ids'].clone() # use the same input as target for language modeling
    else:
        pad_len = input_ids['input_ids'].shape[1] - train_ids.shape[1]
        target_ids = F.pad(train_ids, mode='constant', pad=[0, pad_len, 0, 0], value=0)
        #target_ids = input_ids['input_ids'].clone() 
    # shift the target ids to the left by one and pad with -100
    #target_ids[:, :-1] = target_ids[:, 1:]
    #target_ids[:, -1] = -100
    # zero the gradients
    optimizer.zero_grad()
    # feed the input and target to the model and get the output
    output = model(**input_ids, labels=target_ids)
    # get the loss from the output
    loss = output.loss
    # backpropagate the loss and update the parameters
    loss.backward()
    optimizer.step()
    scheduler.step()
    # accumulate the loss
    total_loss += loss.item()
  # print the average loss per epoch
  logger.info(f'Loss: {total_loss / len(train_dataloader)}')
  logger.info(f'Epoch: {epoch + 1}')
  #dump total loss in a json file
  total_losses.append(total_loss / len(train_dataloader))
#   print(total_losses)

  print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}')

# Save the trained model
model.save_pretrained('trained_model')
#save total losses in a json file
with open('total_losses_opt_'+args.opt+'_lr_'+str(args.lr)+'_bs_'+str(args.b_size)+".json" , 'w') as f:
    json.dump(total_losses, f)



