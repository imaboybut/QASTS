import yaml
import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
from models import get_qat_model
from models.quant import *
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

# if args.dataset == 'audioset':
#     if len(train_loader.dataset) > 2e5:
#         print('scheduler for full audioset is used')
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)
#     else:
#         print('scheduler for balanced audioset is used')
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
#     main_metrics = 'mAP'
#     loss_fn = nn.BCEWithLogitsLoss()
#     warmup = True
# elif args.dataset == 'esc50':
#     print('scheduler for esc-50 is used')
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
#     main_metrics = 'acc'
#     loss_fn = nn.CrossEntropyLoss()
#     warmup = False
# elif args.dataset == 'speechcommands':
#     print('scheduler for speech commands is used')
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
#     main_metrics = 'acc'
#     loss_fn = nn.BCEWithLogitsLoss()
#     warmup = False
# else:
#     raise ValueError('unknown dataset, dataset should be in [audioset, speechcommands, esc50]')
#



args = parser.parse_args()

from sklearn.metrics import accuracy_score, roc_auc_score

def calculate_stats(preds, targets):
    preds = preds.numpy()
    targets = targets.numpy()

    if preds.shape[1] == 1 or (len(preds.shape) == 1):  # Binary classification
        pred_label = (preds > 0.5).astype(int)
        acc = accuracy_score(targets, pred_label)
        auc = roc_auc_score(targets, preds)
    else:  # Multi-class
        pred_label = preds.argmax(axis=1)
        true_label = targets.argmax(axis=1)
        acc = accuracy_score(true_label, pred_label)
        try:
            auc = roc_auc_score(targets, preds, multi_class='ovr')
        except:
            auc = 0.0

    return {'acc': acc, 'auc': auc}
def validate_once(audio_model, val_loader, args):
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_predictions = []
    A_targets = []
    A_loss = []

    with torch.no_grad():
        for audio_input, labels in val_loader:
            audio_input = audio_input.to(device)
            labels = labels.to(device)

            # forward
            audio_output = audio_model(audio_input)

            if args.loss == 'BCE':
                predictions = torch.sigmoid(audio_output).cpu()
                loss = loss_fn(audio_output, labels)
            else:  # 'CE'
                predictions = torch.softmax(audio_output, dim=1).cpu()
                loss = loss_fn(audio_output, torch.argmax(labels, dim=1))

            A_predictions.append(predictions)
            A_targets.append(labels.cpu())
            A_loss.append(loss.cpu())

    audio_output = torch.cat(A_predictions)
    target = torch.cat(A_targets)
    loss = torch.stack(A_loss).mean().item()

    stats = calculate_stats(audio_output, target)

    return stats, loss





# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                  'noise':args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')
    
    float_ckpt_path = "./exp/claude_made/multirate_16k_finetune/models/start_model.pth"
    state_dict = torch.load(float_ckpt_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module."
        else:
            new_state_dict[k] = v
    audio_model.load_state_dict(new_state_dict, strict=True)
    print(audio_model)
    print("    test acc ")

    print("  validation   ...")

    stats, val_loss = validate_once(audio_model, val_loader, args)
    print("FP Model Weight Check (Before QAT):", audio_model.v.blocks[0].attn.qkv.weight.mean())
    print(f"[Pre-QAT] Validation Accuracy: {stats['acc']:.4f}")
    print(f"[Pre-QAT] Validation AUC     : {stats['auc']:.4f}")
    
    # Store FP model performance for comparison
    fp_model_stats = {'acc': stats['acc'], 'auc': stats['auc']}
    
    qat_parser = argparse.ArgumentParser(description='QAT Model Training')
    
    qat_parser.add_argument('--qat_config', type=str, required=True, help='Path to the configuration file')
    
    qat_args = qat_parser.parse_args(['--qat_config', 'qat_config.yml'])

    with open(qat_args.qat_config, 'r') as f:
        qat_config = yaml.safe_load(f)

    for key, value in qat_config.items():
        setattr(qat_args, key, value)
        #qat_args = argparse.Namespace()
    qat_args.__dict__.update(qat_config)
    print("--------------- Loaded Configuration ---------------")
    print(f"Model Type: {qat_args.model_type}")
    print(f"Weight Quantization Bit-width: {qat_args.wq_bitw}")
    print(f"Activation Layer: {qat_args.act_layer}")
    
    print("qat_args")
    print(qat_args)
    print("\n--------------- [Before QAT] Weight Statistics ---------------")
    tensors_to_check = [
        # Patch Embedding
        ('v.patch_embed.proj.weight', audio_model.v.patch_embed.proj.weight),
        
        # Block 0
        ('v.blocks.0.attn.qkv.weight', audio_model.v.blocks[0].attn.qkv.weight),
        ('v.blocks.0.attn.proj.weight', audio_model.v.blocks[0].attn.proj.weight),
        ('v.blocks.0.mlp.fc1.weight', audio_model.v.blocks[0].mlp.fc1.weight),
        ('v.blocks.0.mlp.fc2.weight', audio_model.v.blocks[0].mlp.fc2.weight),

        # Block 1
        ('v.blocks.1.attn.qkv.weight', audio_model.v.blocks[1].attn.qkv.weight),
        ('v.blocks.1.attn.proj.weight', audio_model.v.blocks[1].attn.proj.weight),
        ('v.blocks.1.mlp.fc1.weight', audio_model.v.blocks[1].mlp.fc1.weight),
        ('v.blocks.1.mlp.fc2.weight', audio_model.v.blocks[1].mlp.fc2.weight),

        # Block 2
        ('v.blocks.2.attn.qkv.weight', audio_model.v.blocks[2].attn.qkv.weight),
        ('v.blocks.2.attn.proj.weight', audio_model.v.blocks[2].attn.proj.weight),
        ('v.blocks.2.mlp.fc1.weight', audio_model.v.blocks[2].mlp.fc1.weight),
        ('v.blocks.2.mlp.fc2.weight', audio_model.v.blocks[2].mlp.fc2.weight)
    ]


    for name, tensor in tensors_to_check:
        stats = (f"mean={tensor.mean():.6f}, std={tensor.std():.6f}, "
                 f"min={tensor.min():.6f}, max={tensor.max():.6f}")
        print(f"{name:<30}: {stats}")
    print("------------------------------------------------------------")

    audio_model = get_qat_model(audio_model, qat_args)
    print("\n--------------- [After QAT] Weight Statistics ----------------")
    model_to_check = audio_model.module if isinstance(audio_model, torch.nn.DataParallel) else audio_model
    
    tensors_to_check = [
        # Patch Embedding
        ('v.patch_embed.proj.weight', audio_model.v.patch_embed.proj.weight),
        
        # Block 0
        ('v.blocks.0.attn.qkv.weight', audio_model.v.blocks[0].attn.qkv.weight),
        ('v.blocks.0.attn.proj.weight', audio_model.v.blocks[0].attn.proj.weight),
        ('v.blocks.0.mlp.fc1.weight', audio_model.v.blocks[0].mlp.fc1.weight),
        ('v.blocks.0.mlp.fc2.weight', audio_model.v.blocks[0].mlp.fc2.weight),

        # Block 1
        ('v.blocks.1.attn.qkv.weight', audio_model.v.blocks[1].attn.qkv.weight),
        ('v.blocks.1.attn.proj.weight', audio_model.v.blocks[1].attn.proj.weight),
        ('v.blocks.1.mlp.fc1.weight', audio_model.v.blocks[1].mlp.fc1.weight),
        ('v.blocks.1.mlp.fc2.weight', audio_model.v.blocks[1].mlp.fc2.weight),

        # Block 2
        ('v.blocks.2.attn.qkv.weight', audio_model.v.blocks[2].attn.qkv.weight),
        ('v.blocks.2.attn.proj.weight', audio_model.v.blocks[2].attn.proj.weight),
        ('v.blocks.2.mlp.fc1.weight', audio_model.v.blocks[2].mlp.fc1.weight),
        ('v.blocks.2.mlp.fc2.weight', audio_model.v.blocks[2].mlp.fc2.weight)
    ]


    for name, tensor in tensors_to_check:
        stats = (f"mean={tensor.mean():.6f}, std={tensor.std():.6f}, "
                 f"min={tensor.min():.6f}, max={tensor.max():.6f}")
        print(f"{name:<30}: {stats}")
    print("------------------------------------------------------------\n")



    print("  test acc ")

    print("QAT   validation   ...")

    stats, val_loss = validate_once(audio_model, val_loader, args)

    print(f"[Post-QAT] Validation Accuracy: {stats['acc']:.4f}")
    print(f"[Post-QAT] Validation AUC     : {stats['auc']:.4f}")
    print("\n--- Checking Initialized Quantizer Scaling Factors ('s') ---")
    found_problem = False
    model_to_check = audio_model.module if isinstance(audio_model, nn.DataParallel) else audio_model
    
    for name, module in model_to_check.named_modules():
        if "LsqQuantizer" in module.__class__.__name__:
            scale = getattr(module, 's', None)
            if scale is not None:
                if torch.any(scale.abs() < 1e-9):
                    print(f"!!!!!!!!!!!!!!! PROBLEM FOUND IN MODULE: {name} !!!!!!!!!!!!!!!")
                    print(f"    Scale 's' is zero or near-zero. Mean value: {scale.data.mean().item()}")
                    found_problem = True
                else:
                    print(f"--- OK: {name}, scale mean: {scale.data.mean().item():.6f}")
            else:
                print(f"--- WARNING: {name} has no 's' parameter initialized.")
                found_problem = True
    
    if not found_problem:
        print("--- All scales seem to be initialized to valid non-zero values. ---")
    print("-------------------------------------------------------------------\n")
    # =====================================================================
print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir,exist_ok=True)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
    
