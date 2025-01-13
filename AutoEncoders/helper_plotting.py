import numpy as np
import os
import torch
import matplotlib.pyplot as plt



def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    
    
def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    
def plot_generated_images(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(15, 2.5), n_images=9, modeltype='autoencoder'):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (batch_features, _) in enumerate(data_loader):
        batch_features = batch_features.permute(1,0,2,3,4)
        for features in batch_features:
        
            features = features.to(device)
    
            color_channels = features.shape[1]
            image_height = features.shape[2]
            image_width = features.shape[3]
            
            with torch.no_grad():
                if modeltype == 'autoencoder':
                    decoded_images = model(features)[:n_images]
                elif modeltype == 'VAE':
                    encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
                else:
                    raise ValueError('`modeltype` not supported')
    
            orig_images = features[:n_images]
            break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
                
def plot_latent_space_with_labels(num_classes, data_loader, model, device):
    d = {i:[] for i in range(1,num_classes)}

    with torch.no_grad():
        for i, (batch_features, batch_targets) in enumerate(data_loader):
            batch_features = batch_features.permute(1,0,2,3,4)
            batch_targets = batch_targets.permute(1,0)
            for features in batch_features:
                targets = batch_targets[1]
                
                features = features.to(device)
                targets = targets.to(device)
                
                embedding = model.encoder(features)
    
                for i in range(1,num_classes):
                    if i in targets:
                        mask = targets == i
                        d[i].append(embedding[mask].to('cpu').numpy())

    colors = ["#ff073a","#ff073a","#ff073a","#ff073a","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3",
          "#2984e3","#2984e3","#2984e3","#2984e3","#ff073a","#ff073a","#ff073a","#fce80a","#fce80a","#ff073a",
          "#ff073a","#ff073a","#ff073a","#ff073a","#ff073a","#239124","#239124","#ff073a","#ff073a","#ff073a",
          "#ff073a","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3","#2984e3"]

    for n in range(1,num_classes):
        emp = []
        for i in d[n]:
            for s in i:
                emp.append(s)
        d[n] = np.array(emp)
        flist = []
        for i in d[n]:
            flist.append(i[0])
        slist = []
        for i in d[n]:
            slist.append(i[1])
        
        plt.scatter(
            flist, slist,
            color=colors[n],
            label=f'{n}',
            alpha=0.5)
        
def plot_latent_space_with_labels2(num_classes, data_loader, encoding_fn, device):
    d = {i:[] for i in range(1,num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            batch_features = batch_features.permute(1,0,2,3,4)
            batch_targets = batch_targets.permute(1,0)
            for features in batch_features:

                features = features.to(device)
                targets = targets.to(device)
                
                embedding = encoding_fn(features)

                for i in range(num_classes):
                    if i in targets:
                        mask = targets == i
                        d[i].append(embedding[mask].to('cpu').numpy())

    colors = ["#ff073a","#fe4b03","#e65005","#feb209","#acc2d9","#d4ffff","#0c06f7","#3778bf","#2242c7","#533cc6",
          "#1f6357","#017374","#08787f","#507b9c","#fe2f4a","#9f2305","#c44240","#dee605","#d0e429","#e50000",
          "#840000","#8f1402","#a83c09","#e02f14","#751506","#56ae57","#a8ff04","#d44c37","#b75203","#9a6200",
          "#850e04","#5684ae","#1d5dec","#3a2efe","#26538d","#7bf2da","#107ab0","#247afd","#247afd"]
    
    for n in range(1,num_classes):
        emp = []
        for i in d[n]:
            for s in i:
                emp.append(s)
        d[n] = np.array(emp)
        flist = []
        for i in d[n]:
            flist.append(i[0])
        slist = []
        for i in d[n]:
            slist.append(i[1])
        
        plt.scatter(
            flist, slist,
            color=colors[n],
            label=f'{n}',
            alpha=0.5)
        
def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decoder(rand_features)
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary') 
                
                
def plot_modified_faces(original, diff,
                        diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3.),
                        decoding_fn=None,
                        device=None,
                        figsize=(8, 2.5)):

    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients), 
                             sharex=True, sharey=True, figsize=figsize)
    

    for i, alpha in enumerate(diff_coefficients):
        more = original + alpha*diff
        less = original - alpha*diff
        
        
        if decoding_fn is not None:
            ######################################
            ### Latent -> Original space
            with torch.no_grad():

                if device is not None:
                    more = more.to(device).unsqueeze(0)
                    less = less.to(device).unsqueeze(0)

                more = decoding_fn(more).to('cpu').squeeze(0)
                less = decoding_fn(less).to('cpu').squeeze(0)
            ###################################### 
        
        if not alpha:
            s = 'original'
        else:
            s = f'$\\alpha=${alpha}'
            
        axes[0][i].set_title(s)
        axes[0][i].imshow(more.permute(1, 2, 0))
        axes[1][i].imshow(less.permute(1, 2, 0))
        axes[1][i].axison = False
        axes[0][i].axison = False
        
def plot_multiple_training_losses(losses_list, num_epochs, 
                                  averaging_iterations=100, custom_labels_list=None):

    for i,_ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError('All loss tensors need to have the same number of elements.')
    
    if custom_labels_list is None:
        custom_labels_list = [str(i) for i,_ in enumerate(custom_labels_list)]
    
    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    
    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(range(len(minibatch_loss_tensor)),
                 (minibatch_loss_tensor),
                  label=f'Minibatch Loss{custom_labels_list[i]}')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')

        ax1.plot(np.convolve(minibatch_loss_tensor,
                             np.ones(averaging_iterations,)/averaging_iterations,
                             mode='valid'),
                 color='black')
    
    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(losses_list[i][num_losses:]) for i,_ in enumerate(losses_list)]
    ax1.set_ylim([0, np.max(maxes)*1.5])
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


