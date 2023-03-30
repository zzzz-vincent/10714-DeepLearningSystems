import sys
sys.path.append('./python')
sys.path.append('./apps')
print(sys.path)
import needle as ndl
from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10

device = ndl.cuda()
dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,
         device=device,
         dtype="float32"
         )
model = ResNet9(device=device, dtype="float32")
print(train_cifar10)
train_cifar10(model, dataloader, n_epochs=2, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001, loss_fn=ndl.nn.SoftmaxLoss)
evaluate_cifar10(model, dataloader)