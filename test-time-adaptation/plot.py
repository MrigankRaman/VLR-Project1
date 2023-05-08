import json
import matplotlib.pyplot as plt

with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_Adam_lr_0.001_bs_100.json') as f:
    Adam_100 = json.load(f)
with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_Adam_lr_0.001_bs_1000.json') as f:
    Adam_1000 = json.load(f)
with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_SGD_lr_0.01_bs_100.json') as f:
    SGD_100 = json.load(f)
with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_SGD_lr_0.01_bs_1000.json') as f:
    SGD_1000 = json.load(f)
with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_SGDM_lr_0.01_bs_1000.json') as f:
    SGDM_1000 = json.load(f)
with open('/home/mrigankr/vlr-project/test-time-adaptation/total_losses_opt_SGDM_lr_0.01_bs_100.json') as f:
    SGDM_100 = json.load(f)

plt.plot(Adam_100, label='Adam')
plt.plot(SGD_100, label='SGD')
plt.plot(SGDM_100, label='SGDM')
plt.yscale('log')
plt.legend()
plt.title('Losses for different optimizers with Batch Size 100')
plt.savefig('bs100.png')
plt.clf()
plt.plot(SGD_1000, label='SGD')
plt.plot(SGDM_1000, label='SGDM')
plt.plot(Adam_1000, label='Adam')
plt.yscale('log')
plt.legend()
plt.title('Losses for different optimizers with Batch Size 1000')
plt.savefig('bs1000.png')
