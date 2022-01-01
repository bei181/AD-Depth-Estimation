import numpy as np
import matplotlib.pyplot as plt
import os

def draw_loss(log_file,output,N_img,batch_size):
    
    file = open(log_file, 'r')
    log_info = file.readlines()
    losses, abs_rel, sq_rel, rmse, rmse_log, a1, a2 = list(),list(),list(),list(),list(),list(),list()

    for line in log_info:
        items = line.split()
        if 'loss:' in items:
            try:
                losses.append(float(items[56]))
            except:
                losses.append(float(items[54]))
                
        if 'abs_rel:' in items:
            abs_rel.append(float(items[10].split(',')[0]))
            sq_rel.append(float(items[12].split(',')[0]))
            rmse.append(float(items[14].split(',')[0]))
            rmse_log.append(float(items[16].split(',')[0]))
            a1.append(float(items[18].split(',')[0]))
            a2.append(float(items[20].split(',')[0]))

    num_batch = int(N_img/batch_size)
    epoch = np.arange(0,len(losses))*50/num_batch+1
    plt.figure()
    plt.plot(epoch,losses)
    plt.xlabel('epoch')
    plt.ylabel('total_loss')
    plt.savefig(os.path.join(output,'loss.png'))


    plt.figure(figsize=(20,10))
    plt.subplot(2,3,1);plt.plot(abs_rel);plt.xlabel('epoch');plt.ylabel('abs_rel')
    plt.subplot(2,3,2);plt.plot(sq_rel);plt.xlabel('epoch');plt.ylabel('sq_rel')
    plt.subplot(2,3,3);plt.plot(rmse);plt.xlabel('epoch');plt.ylabel('rmse')
    plt.subplot(2,3,4);plt.plot(rmse_log);plt.xlabel('epoch');plt.ylabel('rmse_log')
    plt.subplot(2,3,5);plt.plot(a1);plt.xlabel('epoch');plt.ylabel('a1')
    plt.subplot(2,3,6);plt.plot(a2);plt.xlabel('epoch');plt.ylabel('a2')
    
    plt.savefig(os.path.join(output,'metrics.png'))


def main():
    N_img = 39810
    batch_size = 2
    log_file = 'log/kitti_fmdepth/20211228_193925.log'
    output = 'log/kitti_fmdepth/'
    draw_loss(log_file,output,N_img,batch_size)

if __name__ == "__main__":
    main()
