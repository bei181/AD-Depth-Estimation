import numpy as np
import matplotlib.pyplot as plt
import os

def draw_fm_loss(log_file,output, N_img, batch_size):
    
    file = open(log_file, 'r')
    log_info = file.readlines()
    losses = list()

    for line in log_info:
        items = line.split()
        if 'loss:' in items:
            try:
                losses.append(float(items[56]))
            except:
                losses.append(float(items[54]))

    num_batch = int(N_img/batch_size)
    epoch = np.arange(0,len(losses))*50/num_batch+1
    plt.figure()
    plt.plot(epoch,losses)
    plt.xlabel('epoch')
    plt.ylabel('total_loss')
    plt.title('FeatDepth-loss')
    plt.savefig(os.path.join(output,'loss_fm.png'))


def draw_auto_loss(log_file, output, N_img, batch_size):
    file = open(log_file, 'r')
    log_info = file.readlines()
    losses = list()

    for line in log_info:
        items = line.split()
        if 'loss:' in items:
            losses.append(float(items[45]))

    num_batch = int(N_img/batch_size)
    epoch = np.arange(0,len(losses))*50/num_batch+1
    plt.figure()
    plt.plot(epoch,losses)
    plt.xlabel('epoch')
    plt.ylabel('total_loss')
    plt.title('Autoencoder-loss')
    plt.savefig(os.path.join(output,'loss_auto.png'))


def main():
    N_img = 10749

    # Draw autoencoder loss
    batch_size = 5
    log_file = './log/my_autoencoder/20211229_193412.log'
    output = './log/my_autoencoder'
    draw_auto_loss(log_file,output,N_img,batch_size)

    # Draw fm loss
    batch_size = 4
    log_file = './log/my_fmdepth_finetune/20211230_185642.log'
    output = './log/my_fmdepth_finetune'
    draw_fm_loss(log_file,output,N_img,batch_size)

    # Draw fm loss
    batch_size = 4
    log_file = './log/my_fmdepth/20211230_183216.log'
    output = './log/my_fmdepth'
    draw_fm_loss(log_file,output,N_img,batch_size)

if __name__ == "__main__":
    main()
