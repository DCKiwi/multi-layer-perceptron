import numpy as np
import mlp 

# Load file 
spam = np.loadtxt('spambase/spambase.data', delimiter=',')

## Attempt to normalise data using maximum
# spam[:,:57] = spam[:,:57]-spam[:,:57].mean(axis=0)
# smax = np.concatenate((spam.max(axis=0)*np.ones((1,58)),np.abs(spam.min(axis=0)*np.ones((1,58)))),axis=0).max(axis=0)
# spam[:,:57] = spam[:,:57]/smax[:57]

# Reshaping target vector
target = np.zeros((np.shape(spam)[0],2))
indices = np.where(spam[:,57] == 0)
target[indices, 0] = 1
indices = np.where(spam[:,57] == 1)
target[indices, 1] = 1

# Randomly order the data
order = range(np.shape(spam)[0])
order = list(range(len(order)))
np.random.shuffle(order)
spam = spam[order,:]
target = target[order,:]

## Second attempt at normalising the data
# rowSum = spam.sum(axis=1)
# newMatrix = spam / rowSum[:, np.newaxis]

# Split into training, validation and test sets
train = spam[0:3450, 0:56]
traint = target[0:3450]
valid = spam[3450:4030, 0:56]
validt = target[3450:4030]
test = spam[4030:4601, 0:56]
testt = target[4030:4601]
 
for i in [1,2,5,10,20]:
  print("------", str(i))

  # Setup network with either 1,2,5,10,20 hidden neuron
  net = mlp.mlp(train,traint,i,outtype='softmax')

  # Train network
  net.earlystopping(train,traint,valid,validt,0.1)

  # Produce confusion matrix
  net.confmat(test,testt)
