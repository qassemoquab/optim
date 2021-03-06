require 'torch'
require 'optim'

n_feature = 3
classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

print'ConfusionMatrix:__init() test'
cm = optim.ConfusionMatrix(#classes, classes)

target = 3
prediction = torch.randn(#classes)

print'ConfusionMatrix:add() test'
cm:add(prediction, target)

batch_size = 8

targets = torch.randperm(batch_size)
predictions = torch.randn(batch_size, #classes)

print'ConfusionMatrix:batchAdd() test'
cm:batchAdd(predictions, targets)
assert(cm.mat:sum() == batch_size + 1, 'missing examples')

print'ConfusionMatrix:updateValids() test'
cm:updateValids()

print'ConfusionMatrix:__tostring__() test'
print(cm)
