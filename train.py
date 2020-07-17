from model import AutoEncoder

if __name__ == '__main__':

    # Train
    if False:
        autoencoder = AutoEncoder(input_shape=(32, 32, 3), latent_dim=64)
        autoencoder.train(train_dir='celeba_data/train', val_dir='celeba_data/val', epochs=20)
    else:
        autoencoder = AutoEncoder(input_shape=(32, 32, 3), latent_dim=64)
        autoencoder.restore_weights()
        autoencoder.reconstruct_samples('test_data')
        autoencoder.generate_samples()
        autoencoder.compute_distance('test_data')