import torch
import sys
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

def create_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
    
    data_trans_train = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_trans_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    class_idx = train_dataset.class_to_idx

    return trainloader, testloader, validloader, class_idx

def prep_model(arch):
    vgg16=''
    alexnet=''
    densenet121=''
    if arch == 'vgg':
        vgg16 = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        densenet121 = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()
        
    model_select = {'vgg':vgg16, 'alexnet':alexnet, 'densenet':densenet121}
    input_size = {'vgg':25088, 'alexnet':9216, 'densenet':1024}
    return model_select[arch], input_size[arch]
    
def create_classifier(model, input_size, hidden_layers, output_size, learning_rate, drop_p=0.5):
    for param in model.parameters():
        param.requires_grad = False
        
    '''classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear('input_size',4096)),
                            ('relu1', nn.ReLU()),
                            ('drop1', nn.Dropout(drop_p)),
                            ('fc2', nn.Linear(4096, ')),
                            ('relu2', nn.ReLU()),
                            ('drop2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(512, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))'''
    hidden_layers = hidden_layers.split(',')
    hidden_layers = [int(x) for x in hidden_layers]
    hidden_layers.append(output_size)
    
    #Take hidden_layer sizes and creates layer size definitions for each hidden_layer size combo
    layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])   
    
    net_layers = OrderedDict()
    
    #Creates hidden layers for each size passed by hidden_layers arg
    for x in range(len(layers)):
        layerid = x + 1
        if x == 0:
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=drop_p)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        else:
            net_layers.update({'relu{}'.format(layerid):nn.ReLU()})
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=drop_p)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        
    net_layers.update({'output':nn.LogSoftmax(dim=1)})
 
    #Define classifier
    classifier = nn.Sequential(net_layers)

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def load_model(filepath):
    trained_model = torch.load(filepath)
    arch = trained_model['arch']
    class_idx = trained_model['class_to_idx']
    
    if arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        load_model = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()
        
    for param in load_model.parameters():
        param.requires_grad = False
        
    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])
    
    return load_model, arch, class_idx

if __name__ == '__main__':
    print('This is run as main.')
    
            
            
          
            

                                                              
    
    
                                         