import copy
import math
import os
import random
import numpy as np

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from tqdm import trange
from examples.images.cifar10.utils_cifar import ema, generate_samples, infiniteloop, setup

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
# from utils_reflow.compute_fid import calculate_fid_ddp
from utils_reflow.utils import seed_everything, generate_image_pairs

# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "icfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 5e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 100001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 10000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", True, help="multi gpu training")
flags.DEFINE_string(
    "master_addr", "localhost", help="master address for Distributed Data Parallel"
)
flags.DEFINE_string("master_port", "12355", help="master port for Distributed Data Parallel")
flags.DEFINE_integer("local_rank", 0, "Local rank of the process in distributed training")
flags.DEFINE_integer("reflow_times", 11, "Reflow times")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    50000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(rank, total_num_gpus, argv):
    seed_everything(42)
    if FLAGS.parallel and total_num_gpus > 1:
        # When using `DistributedDataParallel`, we need to divide the batch
        # size ourselves based on the total number of GPUs of the current node.
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    for reflow_step in range(FLAGS.reflow_times):
        print(
            f"reflow_step: {reflow_step}, lr: {FLAGS.lr}, total_steps: {FLAGS.total_steps}, ema decay: {FLAGS.ema_decay}, save_step: {FLAGS.save_step}",
        )


        # MODELS
        checkpoint_path = "results/reflow/" + f"reflow_{reflow_step}_10.pt"
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')

        net_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(
            rank
        )  # new dropout + bs of 128
        print(f"rank: {rank}")
  
        state_dict = checkpoint["ema_model"]
        try:
            net_model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            net_model.load_state_dict(new_state_dict)
        # net_model.load_state_dict(checkpoint['ema_model'])

        ema_model = copy.deepcopy(net_model)
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
        if FLAGS.parallel:
            net_model = DistributedDataParallel(net_model, device_ids=[rank])
            ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

        # show model size
        model_size = 0
        for param in net_model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        # generate samples 500,000
        device = torch.device(f'cuda:{rank}')
        dataset = generate_image_pairs(net_model, device, rank, total_num_gpus, dataset_size=500000, batch_size=1024)
        
        print(f"generating samples down. Start tring reflow-{reflow_step+1}")

        sampler = DistributedSampler(dataset) if FLAGS.parallel else None
        if FLAGS.parallel:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size_per_gpu,
                sampler=sampler,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                drop_last=True
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size_per_gpu,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                drop_last=True
            )
        
        datalooper = infiniteloop(dataloader)

        # Calculate number of epochs
        steps_per_epoch = math.ceil(len(dataloader.dataset) / (batch_size_per_gpu * (torch.distributed.get_world_size() if FLAGS.parallel else 1)))
        num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)
        print(f"reflow times: {reflow_step}, steps_per_epoch: {steps_per_epoch}, num_epochs: {num_epochs}")
        

        sigma = 0.0
        if FLAGS.model == "otcfm":
            FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif FLAGS.model == "icfm":
            FM = ConditionalFlowMatcher(sigma=sigma)
        elif FLAGS.model == "fm":
            FM = TargetConditionalFlowMatcher(sigma=sigma)
        elif FLAGS.model == "si":
            FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
        else:
            raise NotImplementedError(
                f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
            )

        savedir = FLAGS.output_dir + "reflow" + "/"
        os.makedirs(savedir, exist_ok=True)

        global_step = 0  # to keep track of the global step in training loop

        with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
            for epoch in epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                if sampler is not None:
                    sampler.set_epoch(epoch)

                with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                    for step in step_pbar:
                        global_step += 1

                        optim.zero_grad()
                        x0,  x1 = next(datalooper)
                        x0 = x0.to(rank)
                        x1 = x1.to(rank)

                        # test
                        # test_x1 = x1.view([-1, 3, 32, 32]).clip(-1, 1).detach()
                        # test_x1 = test_x1 / 2 + 0.5
                        # testdir = "./test/"
                        # os.makedirs(testdir, exist_ok=True)
                        # save_image(test_x1.to("cpu"), testdir + f"generated_FM_images_step_{step}.png", nrow=7)
                        # print(f"x0.shape: {x0.shape}, x1.shape: {x1.shape}")
                        # print(f"x1 scale test: {x1.min()}, {x1.max()}")

                        # x0 = torch.randn_like(x1)
                        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                        vt = net_model(t, xt)
                        loss = torch.mean((vt - ut) ** 2)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
                        optim.step()
                        sched.step()
                        ema(net_model, ema_model, FLAGS.ema_decay)  # new

                        # test
                        # print('start test')
                        # testdir = "./test/"
                        # os.makedirs(testdir, exist_ok=True)
                        # generate_samples(
                        #     net_model, FLAGS.parallel, testdir+f"step_{reflow_step}_", global_step//10000, net_="normal"
                        # )
                        # generate_samples(
                        #     ema_model, FLAGS.parallel, testdir+f"step_{reflow_step}_", global_step//10000, net_="ema"
                        # )
                        # print('end test')

                        # sample and Saving the weights
                        if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0:
                        # if global_step % FLAGS.save_step == 0:
                            generate_samples(
                                net_model, FLAGS.parallel, savedir+f"step_{reflow_step}_", global_step//10000, net_="normal"
                            )
                            generate_samples(
                                ema_model, FLAGS.parallel, savedir+f"step_{reflow_step}_", global_step//10000, net_="ema"
                            )
                     
                            torch.save(
                                {
                                    "net_model": net_model.state_dict(),
                                    "ema_model": ema_model.state_dict(),
                                    "sched": sched.state_dict(),
                                    "optim": optim.state_dict(),
                                    "step": global_step,
                                },
                                savedir + f"reflow_{reflow_step+1}_{global_step // 10000}.pt",
                            )
                            # if global_step > 0 and global_step % 150000 == 0:
                            #     device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
                            #     # device = "cuda:3"
                            #     fid_score = calculate_fid_ddp(ema_model, device, 1024, 50000, rank)
                            #     if rank == 0:
                            #         with open(file_path, "a") as file:  
                            #             file.write(f"Iteration {global_step}: FID = {fid_score}\n")
        del net_model
        del ema_model

def main(argv):
    # get world size (number of GPUs)
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch \
#     --nproc_per_node=6 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr="localhost" \
#     --master_port=12355 \
#     train_reflow_10th_cifar10.py



