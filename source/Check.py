from numpy import block
import time
import matplotlib.pyplot as plt

class Check_data:
    @staticmethod
    def check_dataset(dataset):
        #imgs = []
        #titles = ['input', 'label']
        length = dataset.__len__()
        print(length)
        if length > 0:
            print('Dataset is OK')
        print('Checking dataset...')
        for i in range(16):
            input, label = dataset.__getitem__(i)

            # print(i, input.shape, label.shape)
            input = input.squeeze(0).numpy()
            label = label.squeeze(0).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(input, cmap='gray')
            title = axes[0].set_title('Input')
            title.set_fontsize(20)
            axes[1].imshow(label, cmap='gray')
            title = axes[1].set_title('Normal')
            title.set_fontsize(20)
            #plt.savefig(f'./results/check_{i}.png')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            time.sleep(0.5)           
        print('Dataset is checked!')