import torch
import torchvision
import torch.nn as nn
from CompFramework.models.abstract_model_helper import AbstractModelHelper
import CompFramework.datasets.pt.cifar10_dataset as dataset
import CompFramework.models.ptnets.resnet_cifar as net


class ModelHelper(AbstractModelHelper):
  """Model helper for creating a ResNet model for the CIFAR-10 dataset."""

  def __init__(self, args):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__(args)

    # initialize training & evaluation subsets
    self.dataset_train, self.dataset_eval = dataset.get_dataset()
    model = net.resnet20_cifar()
    self.criterion = nn.CrossEntropyLoss().to(args.device)
    self.optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval

  def forward_train(self, inputs):
    """Forward computation at training."""

    pass

  def forward_eval(self, inputs):
    """Forward computation at evaluation."""

    pass

  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""
    pass

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    pass

  @property
  def model_name(self):
    """Model's name."""

    return 'resnet'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'cifar_10'

