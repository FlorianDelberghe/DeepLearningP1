# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import os

import torchvision
from Nets import Net1, Net1bis, Net2, Net2bis, LeNet4, ComparingNet4, LeNet5
import dlc_practical_prologue as prologue
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb

N_PAIRS = 1000

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def progress_bar(pos, total, length=50):
    """
    Retruns a string with the progression of the algorithm
    """
    if pos == total:
        return '[{}]\n'.format('#'*length)

    rel_pos = int(pos/total*length)
    done = '#'*(rel_pos)
    todo = '-'*(length-rel_pos)
    return '[{}]'.format(done+todo)


def create_Net(param):
    """
    Return a Net from one of the availables classes
    """
    if param['net'] == 'Net1':
        return Net1(param)
    elif param['net'] == 'Net1bis':
        return Net1bis(param)
    elif param['net'] == 'Net2':
        return Net2(param)
    elif param['net'] == 'Net2bis':
        return Net2bis(param)
    elif param['net'] == 'LeNet4':
        return LeNet4(param)
    elif param['net'] == 'ComparingNet4':
        return ComparingNet4(param)
    elif param['net'] == 'LeNet5':
        return LeNet5(param)
    else:
        raise NotImplementedError


def train_model(model, train_input, train_target, mini_batch_size, nb_epoch=40, crit=['cross_entropy', 'logistic_loss'], opt='SGD',  progress=True):
    """
    """
    if crit == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
    else:
        raise NotImplementedError

    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if progress:
            print("\r{} loss={:.6E} {}".format(crit.capitalize(),
                                               loss.item(), progress_bar(i+1, nb_epoch)), end='')


def train_model_comparison(model, train_input, train_target, train_classes, mini_batch_size, nb_epoch=40, crit=['cross_entropy', 'soft_margin'], opt='SGD',  progress=True):
    """
    """
    if (crit[0] == 'cross_entropy' and crit[1] == 'soft_margin'):
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
    else:
        raise NotImplementedError

    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-1)
    else:
        raise NotImplementedError

    # replacing 0 by -1 in the target in order to be able to use SoftMarginLoss()
#    train_target[train_target==0]=-1
#    train_target[train_target==1]= 0

    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            x, y, z = model(train_input.narrow(0, b, mini_batch_size))
            loss_x = criterion(x, train_classes.narrow(0, b, mini_batch_size)[:, 0])
            loss_y = criterion(y, train_classes.narrow(0, b, mini_batch_size)[:, 1])
            model.zero_grad()
            (loss_x + loss_y).backward()
            optimizer.step()
        if progress:
            print("\r{} loss_tot={:.6E} {}".format(crit[0].capitalize(
            ), (loss_x + loss_y).item(), progress_bar(i+1, nb_epoch)), end='')

    for i in range(nb_epoch):
        for b in range(0, train_input.size(0), mini_batch_size):
            x, y, z = model(train_input.narrow(0, b, mini_batch_size))
            loss_z = criterion(z, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss_z.backward()
            optimizer.step()
        if progress:
            print("\r{} loss_tot={:.6E} {}".format(
                crit[1].capitalize(), loss_z.item(), progress_bar(i+1, nb_epoch)), end='')


def compute_nb_errors(model, input, target, mini_batch_size):
    """
    Computes the number of errors between a tartget vector and the model prediction usung input
    """
    model.train(False)
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors


def compute_nb_errors_comparison(model, input, classes, target, mini_batch_size):
    """
    Computes the number of errors between a tartget vector and the model prediction usung input
    """
    model.train(False)
    nb_errors_comparison = 0
    nb_errors_recognition = 0

    classes_1 = classes[:, 0]
    classes_2 = classes[:, 1]

    for b in range(0, input.size(0), mini_batch_size):

        predicted_class_1, predicted_class_2, predicted_comparison = model(
            input.narrow(0, b, mini_batch_size))
        # predicted_comparison[predicted_comparison==-1]=0;

        _, predicted_classes_1 = predicted_class_1.data.max(1)
        _, predicted_classes_2 = predicted_class_2.data.max(1)
        _, predicted_comp = predicted_comparison.data.max(1)

        for k in range(mini_batch_size):
            if target.data[b + k] != predicted_comp[k].long():
                nb_errors_comparison += 1
            if classes_1.data[b + k] != predicted_classes_1[k]:
                nb_errors_recognition += 1
            if classes_2.data[b + k] != predicted_classes_2[k]:
                nb_errors_recognition += 1

    return nb_errors_recognition, nb_errors_comparison


def test_model(model, test_input, test_target, test_class):
    """
    Evaluates teh model with classification and comparison
    """
    model.train(False)
    mini_batch_size = 20
    class_err = 0
    for i in [0, 1]:
        class_err += compute_nb_errors(model, Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)),
                                       Variable(test_class[:, i].long()), mini_batch_size)

    pred_class = torch.stack([model(Variable(test_input[:, 0, :, :].view(-1, 1, 14, 14))).data.max(1)[1],
                              model(Variable(test_input[:, 1, :, :].view(-1, 1, 14, 14))).data.max(1)[1]], dim=1)
    comp_err = (compare_digits(pred_class) - test_target.int()).abs().sum().item()

    return class_err, class_err/test_input.size(0)/2, comp_err, comp_err/test_input.size(0)


def test_model_comparison(model, test_input, test_target, test_classes):
    """
    Evaluates teh model with classification and comparison
    """
    model.train(False)
    mini_batch_size = 20

    nb_errors_recognition, nb_errors_comparison = compute_nb_errors_comparison(model, Variable(test_input.view(-1, 2, 14, 14)),
                                                                               Variable(test_classes.long()), Variable(test_target.long()), mini_batch_size)

    return nb_errors_recognition, nb_errors_recognition/test_input.size(0)/2, nb_errors_comparison, nb_errors_comparison/test_input.size(0)


def compare_digits(classes):
    """
    Returns the index of the largest digit
    """
    diff = (classes[:, 1] - classes[:, 0]).sign()
    # gets 0 diff to 1 classification
    return (diff - diff.abs()).div(2).add(1).int()

# %%


def grid_search(param, optimize='epoch', rg=[10, 100], step=10, level=1):
    """
    Does a grid search on a Net optimising for a given parameters
    """
    def get_range_param(rg, step):
        return [i*step+rg[0] for i in range(int((rg[1]-rg[0]) / step)+1)]

    def train_with_set_params(model, nb_epochs, mini_batch_size):
        model.train(True)
        for i in [0, 1]:
            train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)),
                        Variable(train_class[:, i].long()), mini_batch_size, nb_epochs, progress=False)
        model.train(False)

    # Imports the data
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()

    # Sets default
    nb_hidden, nb_epochs, mini_batch_size = param['hidden'], param['epochs'], param['batch_size']

    print("Grid search, Net: {}, optimising: {}, {}->{} step: {}".format(
        param['net'], optimize, rg[0], rg[-1], step), end='\n')

    class_accuracy = []
    comp_accuracy = []

    if optimize == 'epoch':
        model = create_Net(param)
        epochs = get_range_param(rg, step)
        for i, e in enumerate(epochs):
            print("\r{} epochs {}".format(e, progress_bar(2*i, len(epochs)*2-1)), end='')
            # trains the model with the set param
            train_with_set_params(model, e, mini_batch_size)
            print("\r{} epochs {}".format(e, progress_bar(2*i+1, len(epochs)*2-1)), end='')
            # Tests the accuracy
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        plt.figure(figsize=(5, 3))
        plt.plot(epochs, class_accuracy)
        plt.plot(epochs, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('# epochs')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join('figs', "GS{}{}.png".format(param['net'], optimize)))
        plt.show()

    elif optimize == 'hidden_layer_size':
        1
        sizes = get_range_param(rg, step)
        print(sizes)
        for i, s in enumerate(sizes):
            print("\rhd size: {} {}".format(s, progress_bar(2*i, len(sizes)*2-1)), end='')
            # Creates modle with s hidden layers
            param['hidden'] = s
            model = create_Net(param)
            train_with_set_params(model, nb_epochs, mini_batch_size)
            print("\rhd size: {} {}".format(s, progress_bar(2*i+1, len(sizes)*2-1)), end='')
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        plt.figure(figsize=(5, 3))
        plt.plot(sizes, class_accuracy)
        plt.plot(sizes, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('# hidden layers')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join('figs', "GS{}{}.png".format(param['net'], optimize)))
        plt.show()

    elif optimize == 'mini_batch_size':
        model = create_Net(param)
        for i, s in enumerate(rg):
            print("\rmb size: {} {}".format(s, progress_bar(2*i, len(rg)*2-1)), end='')
            train_with_set_params(model, nb_epochs, s)
            print("\rmb size: {} {}".format(s, progress_bar(2*i+1, len(rg)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        plt.figure(figsize=(5, 3))
        plt.plot(rg, class_accuracy)
        plt.plot(rg, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('Mini batch size')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join('figs', "GS{}{}.png".format(param['net'], optimize)))
        plt.show()

    elif optimize == 'drop_proba':
        probas = get_range_param(rg, step)
        for i, p in enumerate(probas):
            param['drop_proba'][level] = p
            model = create_Net(param)
            print("\rdrop proba level {}: {}".format(
                level, progress_bar(2*i, len(probas)*2-1)), end='')
            train_with_set_params(model, nb_epochs, mini_batch_size)
            print("\rdrop proba level {}: {}".format(
                level, progress_bar(2*i+1, len(probas)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model(model, test_input, test_target, test_class)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        plt.figure(figsize=(5, 3))
        plt.plot(probas, class_accuracy)
        plt.plot(probas, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('Drop proba @ level {}'.format(level))
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join('figs', "GS{}{}.png".format(param['net'], optimize)))
        plt.show()

    else:
        raise NotImplementedError


def grid_search_comparison(param, optimize='epoch', rg=[10, 100], step=10, level=1):
    """
    Does a grid search on a Net optimising for a given parameters
    """
    def get_range_param(rg, step):
        return [i*step+rg[0] for i in range(int((rg[1]-rg[0]) / step)+1)]

    def train_with_set_params(model, nb_epochs, mini_batch_size):
        model.train(True)

        train_model_comparison(model, Variable(train_input.view(-1, 2, 14, 14)), Variable(
            train_target.long()), Variable(train_classes.long()), param['batch_size'], param['epochs'])
        model.train(False)

    # Imports the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(N_PAIRS)
    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()

    train_input, train_target = train_input.to(device), train_target.to(device)
    test_input, test_target = test_input.to(device), test_target.to(device)
    train_classes, test_classes = train_classes.to(device), test_classes.to(device)
    # Sets default
    nb_hidden, nb_epochs, mini_batch_size = param['hidden'], param['epochs'], param['batch_size']

    print("Grid search, Net: {}, optimising: {}, {}->{} step: {}".format(
        param['net'], optimize, rg[0], rg[-1], step), end='\n')

    class_accuracy = []
    comp_accuracy = []

    if optimize == 'epoch':
        model = create_Net(param)
        model.to(device)
        epochs = get_range_param(rg, step)
        for i, e in enumerate(epochs):
            print("\r{} epochs {}".format(e, progress_bar(2*i, len(epochs)*2-1)), end='')
            # trains the model with the set param
            train_with_set_params(model, e, mini_batch_size)
            print("\r{} epochs {}".format(e, progress_bar(2*i+1, len(epochs)*2-1)), end='')
            # Tests the accuracy
            metrics = test_model_comparison(model, test_input, test_target, test_classes)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        print("==> Best param according to class_accuracy = {}".format(
            epochs[np.argmin(class_accuracy)]))
        print("==> Best param according to comp_accuracy = {}".format(
            epochs[np.argmin(comp_accuracy)]), end='\n\n')
        plt.figure(figsize=(5, 3))
        plt.plot(epochs, class_accuracy)
        plt.plot(epochs, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('# epochs')
        plt.ylabel('Accuracy')
        plt.savefig('fig/GS_{}_{}'.format(param['net'], optimize)+'.tif', format="tif")
        plt.show()

    elif optimize == 'hidden_layer_size':
        sizes = get_range_param(rg, step)
        class_accuracy_hidden = np.empty((len(sizes), len(sizes)))
        comp_accuracy_hidden = np.empty((len(sizes), len(sizes)))
        print(sizes)
        for i, s in enumerate(sizes):
            for j, t in enumerate(sizes):
                print("\rhd size 1: {} {}".format(s, progress_bar(2*i, len(sizes)*2-1)), end='')
                print("\rhd size 2: {} {}".format(t, progress_bar(2*j, len(sizes)*2-1)), end='')
                # Creates modle with s hidden layers
                param['hidden'][0] = s
                param['hidden'][1] = t
                model = create_Net(param)
                model.to(device)
                train_with_set_params(model, nb_epochs, mini_batch_size)
                print("\rhd size: {} {}".format(s, progress_bar(2*i+1, len(sizes)*2-1)), end='')
                metrics = test_model_comparison(model, test_input, test_target, test_classes)
                print(metrics, end='\n')
                class_accuracy_hidden[i][j] = metrics[1]
                comp_accuracy_hidden[i][j] = metrics[3]

                cdict = {'red': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
                         'green': ((0., 1, 1), (0.05, 1, 1), (0.11, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
                         'blue': ((0., 1, 1), (0.05, 1, 1), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}

                my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict)

        argmin_class = np.where(class_accuracy_hidden == np.min(class_accuracy_hidden))
        argmin_comp = np.where(comp_accuracy_hidden == np.min(comp_accuracy_hidden))
        print("==> Best hidden sizes according to class_accuracy = [{}, {}]".format(
            sizes[argmin_class[0][0]], sizes[argmin_class[1][0]]), end='\n')
        print("==> Best hidden sizes according to comp_accuracy = [{}, {}]".format(
            sizes[argmin_comp[0][0]], sizes[argmin_comp[1][0]]), end='\n\n')

        plt.figure()
        plt.pcolor(class_accuracy_hidden, cmap=my_cmap)
        plt.colorbar()
        plt.title("class_accuracy_hidden")
        plt.show()
        plt.savefig(
            'fig/GS_classification_{}_{}'.format(param['net'], optimize)+'.tif', format="tif")
        plt.figure()
        plt.pcolor(comp_accuracy_hidden, cmap=my_cmap)
        plt.colorbar()
        plt.title("comp_accuracy_hidden")
        plt.show()
        plt.savefig('fig/GS_comparison_{}_{}'.format(param['net'], optimize)+'.tif', format="tif")

    elif optimize == 'mini_batch_size':
        model = create_Net(param)
        model.to(device)
        for i, s in enumerate(rg):
            print("\rmb size: {} {}".format(s, progress_bar(2*i, len(rg)*2-1)), end='')
            train_with_set_params(model, nb_epochs, s)
            print("\rmb size: {} {}".format(s, progress_bar(2*i+1, len(rg)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model_comparison(model, test_input, test_target, test_classes)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        argmin_class = np.argmin(class_accuracy)
        argmin_comp = np.argmin(comp_accuracy)
        print("==> Best mini batch size according to class_accuracy = {}".format(
            rg[argmin_class]), end='\n')
        print("==> Best mini batch size according to comp_accuracy = {}".format(
            rg[argmin_comp]), end='\n\n')

        plt.figure(figsize=(5, 3))
        plt.plot(rg, class_accuracy)
        plt.plot(rg, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('Mini batch size')
        plt.ylabel('Accuracy')
        plt.savefig('fig/GS_{}_{}'.format(param['net'], optimize)+'.tif', format="tif")
        plt.show()

    elif optimize == 'drop_proba':
        probas = get_range_param(rg, step)
        for i, p in enumerate(probas):
            param['drop_proba'][level] = p
            model = create_Net(param)
            model.to(device)
            print("\rdrop proba level {}: {}".format(
                level, progress_bar(2*i, len(probas)*2-1)), end='')
            train_with_set_params(model, nb_epochs, mini_batch_size)
            print("\rdrop proba level {}: {}".format(
                level, progress_bar(2*i+1, len(probas)*2-1)), end='')
            # training batch size is irrelevant for the test
            metrics = test_model_comparison(model, test_input, test_target, test_classes)
            class_accuracy.append(metrics[1])
            comp_accuracy.append(metrics[3])

        argmin_class = np.argmin(class_accuracy)
        argmin_comp = np.argmin(comp_accuracy)
        print("==> Best drop_proba according to class_accuracy = {}".format(
            probas[argmin_class]), end='\n')
        print("==> Best drop_proba according to comp_accuracy = {}".format(
            probas[argmin_comp]), end='\n')

        plt.figure(figsize=(5, 3))
        plt.plot(probas, class_accuracy)
        plt.plot(probas, comp_accuracy)
        plt.legend(['Classification accuracy', 'Comparison accuracy'])
        plt.xlabel('Drop proba @ level {}'.format(level))
        plt.ylabel('Accuracy')
        plt.savefig('fig/GS_{}_{}'.format(param['net'], optimize)+'.tif', format="tif")
        plt.show()

    else:
        raise NotImplementedError


# %%
def test_param(param):

    print("Testing...", end='')
    for k in param.keys():
        print(" {}: {} |".format(k, param[k]), end='')
    print('')

    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)

    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()

    model = create_Net(param)
    model.train(True)
    for i in [0, 1]:
        train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)),
                    Variable(train_class[:, i].long()), param['batch_size'], param['epochs'])

    model.train(False)
    nb_test_errors = 0
    for i in [0, 1]:
        nb_test_errors += compute_nb_errors(model, Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)),
                                            Variable(test_class[:, i].long()), 20)

    print('Classification test error: {:0.2f}% {:d}/{:d}'.format(nb_test_errors / test_input.size(0) / 2*100,
                                                                 nb_test_errors, 2*test_input.size(0)))

    pred_class = torch.stack([model(Variable(test_input[:, 0, :, :].view(-1, 1, 14, 14))).data.max(1)[1],
                              model(Variable(test_input[:, 1, :, :].view(-1, 1, 14, 14))).data.max(1)[1]], dim=1)

    print("Comparison error: {:0.2f}% {:d}/{:d}".format((compare_digits(pred_class) - test_target.int()).abs().sum().item() / test_target.size(0)*100,
                                                        (compare_digits(pred_class) -
                                                         test_target.int()).abs().sum().item(),
                                                        test_input.size(0)))


def test_param_comparison(param):

    print("Testing...", end='')
    for k in param.keys():
        print(" {}: {} |".format(k, param[k]), end='')
    print('')

    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(N_PAIRS)

    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()

    model = create_Net(param)
    model.train(True)

    train_model_comparison(model, Variable(train_input.view(-1, 2, 14, 14)),
                           Variable(train_target.long()), Variable(train_classes.long()), param['batch_size'], param['epochs'])

    model.train(False)

    nb_errors_recognition, nb_errors_comparison = compute_nb_errors_comparison(model, Variable(test_input.view(-1, 2, 14, 14)),
                                                                               Variable(test_classes.long()), Variable(test_target.long()), param['batch_size'])

    print('Recognition test error: {:0.2f}% {:d}/{:d}'.format(nb_errors_recognition / test_input.size(0) / 2*100,
                                                              nb_errors_recognition, 2*test_input.size(0)))

    print("Comparison error: {:0.2f}% {:d}/{:d}".format(nb_errors_comparison / test_input.size(0) * 100,
                                                        nb_errors_comparison, test_input.size(0)))


def boosted_Net(net_dicts):
    """
    Trains multiple nets and boosts the results to have better perfs
    """
    print("Creating boosted prediction with {} nets".format(len(net_dicts)))
    train_input, train_target, train_class, test_input, test_target, test_class = \
        prologue.generate_pair_sets(N_PAIRS)

    # Normalization of the data
    train_input /= train_input.max()
    test_input /= test_input.max()

    pred_tr = []
    pred_te = []
    for i, dic in enumerate(net_dicts):
        print("Net {}/{}: {}".format(i+1, len(net_dicts), dic['net']))
        # Sets a different seed for each net
        torch.manual_seed(dic['seed'])
        model = create_Net(dic)
        # Training each of the models
        model.train(True)
        for i in [0, 1]:
            train_model(model, Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)),
                        Variable(train_class[:, i].long()), dic['batch_size'], dic['epochs'])
        model.train(False)

        print(test_model(model, test_input, test_target, test_class))

        output_tr = []
        output_te = []
        for i in [0, 1]:
            out = model(Variable(train_input[:, i, :, :].view(-1, 1, 14, 14)))
            output_tr.append(out.data.max(1)[1])

            out = model(Variable(test_input[:, i, :, :].view(-1, 1, 14, 14)))
            output_te.append(out.data.max(1)[1])

        pred_tr.append(torch.cat(output_tr))
        pred_te.append(torch.cat(output_te))

    # .reshape((train_class.size(0)*2, 1))
    target = torch.cat([train_class[:, 0], train_class[:, 1]])
    boost_model = xgb.XGBClassifier().fit(torch.stack(pred_tr, 1), target)
    boost_pred = boost_model.predict(torch.stack(pred_te, 1))

    pred_classes = torch.Tensor(np.stack([boost_pred[:1000], boost_pred[1000:]], axis=1))
    print("Classification error w/ boosting: {:.4}%".format((boost_pred != torch.cat(
        [test_class[:, 0], test_class[:, 1]]).numpy()).sum() / boost_pred.shape[0]*100))
    print("Comparison error w/ boosting: {:.4}%\n".format(
        (compare_digits(pred_classes) - test_target.int()).abs().sum().item() / test_target.size(0)*100))


def main():
    net2 = {"net": 'Net2', "hidden": 120, "epochs": 80, "batch_size": 20, "pool": 'max',
            "activation": 'relu', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": 0}
    net4 = {"net": 'LeNet4', "hidden": 350, "epochs": 30, "batch_size": 20, "pool": 'max',
            "activation": 'tanh', "drop_proba": [0.0, 0.0, 0.0, 0.0, 0.1], "seed": 1}

    net5 = {"net": 'LeNet5', "hidden": 250, "epochs": 30, "batch_size": 20, "pool": 'avg',
            "activation": 'tanh', "drop_proba": [0.0, 0.0, 0.2, 0.25, 0.15], "seed": 0}
    net5 = {"net": 'LeNet5', "hidden": 120, "epochs": 30, "batch_size": 20, "pool": 'max',
            "activation": 'relu', "drop_proba": [0.0, 0.0, 0.0, 0.0], "seed": 0}

#    grid_search(net4 ,optimize='hidden_layer_size', rg=[10, 80], step=10)

#    test_param(net4)

#    nets = []
#    for i in range(3):
#        nets.append({"net": 'LeNet4', "hidden": 350, "epochs": 30, "batch_size": 20, "pool": 'max', "activation": 'tanh', "drop_proba": [0.0, 0.0, 0.0, 0.0, 0.1], "seed": i})
#    boosted_Net(nets)

#    net1bis =  {"net": 'Net1bis', "hidden": 120, "epochs": 80, "batch_size": 20, "pool": 'max', "activation": 'relu', "seed": 0}
#    net1 =  {"net": 'Net1', "hidden": 120, "epochs": 80, "batch_size": 20, "pool": 'max', "activation": 'relu', "seed": 0}
#
#    net2bis =  {"net": 'Net2bis', "hidden": 120, "epochs": 80, "batch_size": 20, "pool": 'max', "activation": 'relu', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": 0}
# %%
    comparingNet4 = {"net": 'ComparingNet4', "hidden": [120, 50], "epochs": 30, "batch_size": 20,
                     "pool": 'max', "activation": 'tanh', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": 1}

    # GS1_CompNet4_29.04.2019
    # grid_search_comparison(comparingNet4, optimize='epoch', rg=[10, 200], step=10, level=1)
    # grid_search_comparison(comparingNet4, optimize='hidden_layer_size',
    #                        rg=[10, 120], step=10, level=1)
    # grid_search_comparison(comparingNet4, optimize='mini_batch_size',
    #                        rg=[10, 250], step=10, level=1)
    # grid_search_comparison(comparingNet4, optimize='drop_proba', rg=[0.001, 1], step=10, level=1)
# %%
    comparingNet4 = {"net": 'ComparingNet4', "hidden": [55, 90], "epochs": 150, "batch_size": 250,
                     "pool": 'max', "activation": 'tanh', "drop_proba": [0.001, 0.001, 0.001, 0.001], "seed": 1}

    test_param_comparison(comparingNet4)
# %%
    comparingNet4 = {"net": 'ComparingNet4', "hidden": [60, 90], "epochs": 150, "batch_size": 250,
                     "pool": 'max', "activation": 'tanh', "drop_proba": [0.05, 0.05, 0.5, 0.2], "seed": 1}

    test_param_comparison(comparingNet4)
# %%


if __name__ == '__main__':
    main()
