import torch
from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self, latent_size=512, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            self.classifier = nn.Sequential(nn.Linear(latent_size, 1))
        else:
            self.classifier = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                            nn.LeakyReLU(0.1),nn.Linear(hidden_size,1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.classifier(x)
        prob = self.sigmoid(out)
        return prob, out


class MulticlassClassifier(nn.Module):
    def __init__(self, latent_size=512, n_classes=8, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            self.classifier = nn.Sequential(nn.Linear(latent_size, n_classes))
        else:
            self.classifier = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                            nn.LeakyReLU(0.1),nn.Linear(hidden_size,n_classes))
        self.softmax = nn.Softmax()
    def forward(self, x):
        out = self.classifier(x)
        prob = self.softmax(out)
        return prob, out



class MultiLabelClassifier(nn.Module):
    def __init__(self, attributes=None, n_attributes=0, n_shared_layers=0, hidden_size=None, latent_size=512, multiclass_classes=None):
        super().__init__()
        assert n_attributes > 0
        self.attributes = attributes.split(',')
        if n_shared_layers > 0:
            shared = []
            shared.append(nn.Linear(latent_size,latent_size))
            shared.append(nn.LeakyReLU(0.1))
            for i in range(1,n_shared_layers):
                shared.append(nn.Linear(latent_size,latent_size))
                shared.append(nn.LeakyReLU(0.1))
        else:
            shared = [nn.Identity()]

        self.n_branches = n_attributes
        if multiclass_classes is not None:
            self.multiclass_classes = [int(c) for c in multiclass_classes.split(',')]
        else:
            self.multiclass_classes = None

        self.shared = nn.Sequential(*shared)

        classifier_dict = dict()
        for i in range(self.n_branches):
            if multiclass_classes is not None and self.multiclass_classes[i]>1:
                classifier_dict[str(i)] = MulticlassClassifier(hidden_size=hidden_size, n_classes=self.multiclass_classes[i])
            else:
                classifier_dict[str(i)] = BinaryClassifier(hidden_size=hidden_size)
        self.classifiers = nn.ModuleDict(classifier_dict)

    def forward(self, x):
        """
        x is of shape b_size x latent_dim x num_branches
        output is of shape b_size x 1 x num_branches
        at inference time: repeat x num_branches times
        """
        if len(x.shape)==2:
            x = x.unsqueeze(2).repeat(1,1,self.n_branches)
        shared = torch.cat([self.shared(x[:,:,i]).unsqueeze(2) for i in range(self.n_branches)], dim=2)
        out = []
        for i in range(self.n_branches):
            out.append(self.classifiers[str(i)](shared[:,:,i])[0].unsqueeze(2))
        out = torch.cat(out, dim=2)
        #calculate full_output for validation
        full_out = []
        full_in = torch.cat([shared[:,:,i] for i in range(self.n_branches)],dim=0)
        for i in range(self.n_branches):
            full_out.append(self.classifiers[str(i)](full_in)[1].unsqueeze(2))
        full_out = torch.cat(full_out, dim=2)
        return out, full_out


