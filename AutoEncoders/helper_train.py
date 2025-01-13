from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss_classifier
from helper_evaluate import compute_epoch_loss_autoencoder

import time
import torch
import torch.nn.functional as F
import torchvision
import torch.autograd


def train_classifier_simple_v1(num_epochs, model, optimizer, device,
                               train_loader, valid_loader=None,
                               loss_fn=None, logging_interval=100,
                               skip_epoch_stats=False):

    log_dict = {'train_loss_per_batch': [],
                'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss_classifier(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f' % (
                      epoch+1, num_epochs, train_acc, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
                log_dict['train_acc_per_epoch'].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss_classifier(
                        model, valid_loader, loss_fn, device)
                    print('***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f' % (
                          epoch+1, num_epochs, valid_acc, valid_loss))
                    log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                    log_dict['valid_acc_per_epoch'].append(valid_acc.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    return log_dict


def train_autoencoder_v1(num_epochs, model, optimizer, device,
                         train_loader, loss_fn=None,
                         logging_interval=100,
                         skip_epoch_stats=False,
                         save_model=None):

    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (batch_features, _) in enumerate(train_loader):
            batch_features = batch_features.permute(1,0,2,3,4)
            for features in batch_features:
                features = features.to(device)

                # FORWARD AND BACK PROP
                logits = model(features)
                loss = loss_fn(logits, features)
                optimizer.zero_grad()

                loss.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                log_dict['train_loss_per_batch'].append(loss.item())

                if not batch_idx % logging_interval:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                        % (epoch+1, num_epochs, batch_idx,
                            len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return log_dict

def train_vae_v1(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100, 
                 skip_epoch_stats=False,
                 reconstruction_term_weight=1,
                 save_model=None):

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (batch_features, _) in enumerate(train_loader):
            batch_features = batch_features.permute(1,0,2,3,4)
            for features in batch_features:

                features = features.to(device)

                # FORWARD AND BACK PROP
                encoded, z_mean, z_log_var, decoded = model(features)

                # total loss = reconstruction loss + KL divergence
                # kl_divergence = (0.5 * (z_mean**2 +
                #                         torch.exp(z_log_var) - z_log_var - 1)).sum()
                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                        - z_mean**2 
                                        - torch.exp(z_log_var), 
                                        axis=1) # sum over latent dimension

                batchsize = kl_div.size(0)
                kl_div = kl_div.mean() # average over batch dimension
        
                pixelwise = loss_fn(decoded, features, reduction='none')
                pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
                pixelwise = pixelwise.mean() # average over batch dimension
                
                loss = reconstruction_term_weight*pixelwise + kl_div
                
                optimizer.zero_grad()

                loss.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                log_dict['train_combined_loss_per_batch'].append(loss.item())
                log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
                log_dict['train_kl_loss_per_batch'].append(kl_div.item())
                
                if not batch_idx % logging_interval:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                        % (epoch+1, num_epochs, batch_idx,
                            len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_combined_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


def update_running_avg(running_avg, new_value, decay=0.99):
    return decay * running_avg + (1 - decay) * new_value

def train_gan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                 latent_dim, device, train_loader, loss_fn=None,
                 logging_interval=100, save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    fixed_noise_spherical = fixed_noise / fixed_noise.norm(dim=1, keepdim=True)

    start_time = time.time()
    
    running_avg_gen_loss = 0
    running_avg_discr_loss = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (batch_features, _) in enumerate(train_loader):
            batch_features = batch_features.permute(1,0,2,3,4)
            for real_images in batch_features:
                real_images = real_images.to(device)
                # Normalize to [0, 1]
                min_val = real_images.amin(dim=(2, 3), keepdim=True)
                max_val = real_images.amax(dim=(2, 3), keepdim=True)
                norm_real_images = (real_images - min_val) / (max_val - min_val)
                
                # Normalize to [-1, 1]
                norm_real_images = norm_real_images * 2 - 1
                batch_size = norm_real_images.size(0)
    
                # Generate fake images
                noise = torch.randn(batch_size, latent_dim, device=device)  # Shape: (batch_size, latent_dim)
                spherical_noise = noise / noise.norm(dim=1, keepdim=True)  # Normalize to unit sphere
                # Pass spherical_noise with correct shape to the generator
                fake_images = model.generator_forward(spherical_noise)
                
                # --------------------------
                # Train Discriminator
                # --------------------------
                optimizer_discr.zero_grad()
                discr_pred_real = model.discriminator_forward(norm_real_images).view(-1)
                discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)

                real_loss = loss_fn(discr_pred_real, torch.ones(batch_size, device=device))
                fake_loss = loss_fn(discr_pred_fake, torch.zeros(batch_size, device=device))
                discr_loss = (real_loss + fake_loss) / 2
    
                discr_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=5.0)
                optimizer_discr.step()
    
                # --------------------------
                # Train Generator
                # --------------------------
                optimizer_gen.zero_grad()
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, torch.ones(batch_size, device=device))
    
                gener_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=5.0)
                optimizer_gen.step()
    
                # Update running averages
                running_avg_gen_loss = update_running_avg(running_avg_gen_loss, gener_loss.item())
                running_avg_discr_loss = update_running_avg(running_avg_discr_loss, discr_loss.item())

                # --------------------------
                # Logging
                # --------------------------
                if batch_idx % logging_interval == 0:
                    log_dict['train_generator_loss_per_batch'].append(running_avg_gen_loss)
                    log_dict['train_discriminator_loss_per_batch'].append(running_avg_discr_loss)
    
                    acc_real = (torch.sigmoid(discr_pred_real) > 0.5).float().mean().item() * 100
                    acc_fake = (torch.sigmoid(discr_pred_fake) <= 0.5).float().mean().item() * 100
    
                    print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                          f'Gen Loss (Avg): {running_avg_gen_loss:.4f} | Dis Loss (Avg): {running_avg_discr_loss:.4f} | '
                          f'Real Acc: {acc_real:.2f}% | Fake Acc: {acc_fake:.2f}%')

        # Save images for evaluation
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise_spherical).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict



def train_gan_v2(num_epochs, model, optimizer_gen, optimizer_discr, 
                 latent_dim, device, train_loader, loss='regular',
                 logging_interval=100, 
                 save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if loss == 'regular':
        loss_fn = F.binary_cross_entropy_with_logits
    elif loss == 'wasserstein':
        def loss_fn(y_pred, y_true):
            return -torch.mean(y_pred * y_true)
    else:
        raise ValueError('This loss is not supported.')

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
            
            if loss == 'regular':
                fake_labels = torch.zeros(batch_size, device=device) # fake label = 0
            elif loss == 'wasserstein':
                fake_labels = -real_labels # fake label = -1    
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
            
            if loss == 'wasserstein':
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

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


def train_wgan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                  latent_dim, device, train_loader,
                  discr_iter_per_generator_iter=5,
                  logging_interval=100, 
                  gradient_penalty=False,
                  gradient_penalty_weight=10,
                  save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    # Batch of latent (noise) vectors for
    # evaluating / visualizing the training progress
    # of the generator
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device) # format NCHW

    start_time = time.time()
    
    
    skip_generator = 1
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
            
            fake_labels = -real_labels # fake label = -1    
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

            ###################################################
            # gradient penalty
            if gradient_penalty:

                alpha = torch.rand(batch_size, 1, 1, 1).to(device)

                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True

                discr_out = model.discriminator_forward(interpolated)

                grad_values = torch.ones(discr_out.size()).to(device)
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                discr_loss += gp_penalty_term
                
                log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())
            #######################################################
            
            discr_loss.backward()

            optimizer_discr.step()
            
            # Use weight clipping (standard Wasserstein GAN)
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            
            if skip_generator <= discr_iter_per_generator_iter:
                
                # --------------------------
                # Train Generator
                # --------------------------

                optimizer_gen.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                optimizer_gen.step()
                
                skip_generator += 1
                
            else:
                skip_generator = 1
                gener_loss = torch.tensor(0.)

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