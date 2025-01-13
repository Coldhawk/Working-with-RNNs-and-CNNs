import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import torchvision
import tensorflow as tf
import cv2

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

def train_gan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                 latent_dim, device, train_loader, loss_fn=None,
                 logging_interval=100, 
                 save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits

    # Batch of latent (noise) vectors for
    # evaluating / visualizing the training progress
    # of the generator
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device) # format NCHW

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size, device=device) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))

        ### Save images for evaluation
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

def plot_latent_space_with_labels2(num_classes, data_loader, encoding_fn, device):
    d = {i:[] for i in range(1,num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

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
    for i in range(1,num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i],
            label=f'{i}',
            alpha=0.5)
        
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, paths, targets, image_size=(224, 224), batch_size=64, 
        shuffle=True, transforms=None, preprocess=None,
    ):
        # the list of paths to files
        self.paths = paths
        # the list with the true labels of each file
        self.targets = targets
        # images size
        self.image_size = image_size
        # batch size (the number of images)
        self.batch_size = batch_size
        # if we need to shuffle order of files
        # for validation we don't need to shuffle, for training - do
        self.shuffle = shuffle
        # Augmentations for our images. It is implemented with albumentations library
        self.transforms = transforms
        # Preprocess function for the pretrained model. 
        # CHANGE IT IF USING OTHER THAN MOBILENETV2 MODEL
        self.preprocess = preprocess
        
        # Call function to create and shuffle (if needed) indices of files
        self.on_epoch_end()

        
    def on_epoch_end(self):
        # This function is called at the end of each epoch while training
        
        # Create as many indices as many files we have
        self.indexes = np.arange(len(self.paths))
        # Shuffle them if needed
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        # We need that this function returns the number of steps in one epoch
        
        # How many batches we have
        return len(self.paths) // self.batch_size
    
    
    def __getitem__(self, index):
        # This function returns batch of pictures with their labels
        
        # Take in order as many indices as our batch size is
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Take image file paths that are included in that batch
        batch_paths = [self.paths[k] for k in indexes]
        # Take labels for each image
        batch_y = [self.targets[k] - 1 for k in indexes]
        batch_X = []
        for i in range(self.batch_size):
            # Read the image
            img = cv2.imread(batch_paths[i])
            # Resize it to needed shape
            img = cv2.resize(img, self.image_size)
            # Convert image colors from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Apply transforms (see albumentations library)
            if self.transforms:
                img = self.transforms(image=img)["image"]
            # Apply preprocess
            if self.preprocess:
                img = self.preprocess(img)
            
            batch_X.append(img)
            
        return torch.from_numpy(np.array(batch_X)), torch.from_numpy(np.array(batch_y))