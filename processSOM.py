import pylab as pl
import numpy as np
import kmeansnet
import som

spam = np.loadtxt('spambase/spambase.data', delimiter=',')
spam[:,:57] = spam[:,:57]-spam[:,:57].mean(axis=0)
smax = np.concatenate((spam.max(axis=0)*np.ones((1,58)),np.abs(spam.min(axis=0)*np.ones((1,58)))),axis=0).max(axis=0)
spam[:,:57] = spam[:,:57]/smax[:57]

target = spam[:,57]

order = range(np.shape(spam)[0])
np.random.shuffle(order)
spam = spam[order,:]
target = target[order]

train = spam[0:3449, 0:56]
traint = target[0:3449]
valid = spam[3450:4029, 0:56]
validt = target[3450:4029]
test = spam[4030:4600, 0:56]
testt = target[4030:4600]

net = kmeansnet.kmeans(3,train)
net.kmeanstrain(train)
cluster = net.kmeansfwd(test)

net = som.som(6,6,train)
net.somtrain(train,400)

best = np.zeros(np.shape(train)[0],dtype=int)
for i in range(np.shape(train)[0]):
    best[i],activation = net.somfwd(train[i,:])

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(traint == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(traint == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(traint == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.figure(2)

best = np.zeros(np.shape(test)[0],dtype=int)
for i in range(np.shape(test)[0]):
    best[i],activation = net.somfwd(test[i,:])

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(testt == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(testt == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(testt == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.show()

