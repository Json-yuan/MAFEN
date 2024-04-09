"""
    训练
"""
import argparse
import os
import logging
import datetime

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.scheduler import create_scheduler
from tqdm import tqdm
import utils.JutinLoss as JutinLoss
from utils import cutdataset
from data.custom_data import MyCustomDataset, collate_classification
import SwinVit_1
import utils
device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')
print("using {} device.".format(device))
print(torchvision.__version__)
print(torch.__version__)
def get_args_parser(root, dataset_path):
    """

    :param root: 项目路径
    :param dataset_path: 数据集路径
    :return:
    """
    parser = argparse.ArgumentParser("Net train and evaluation", add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    # DataSet
    parser.add_argument('--dataset_path', default=dataset_path)
    parser.add_argument('--dataset', default='NWPU', help='Which dataset')
    parser.add_argument('--train_ratio', default=0.2, type=float)
    # UC_Merced  NWPU   AID     RSSDIVCS
    parser.add_argument('--log_dir', default=os.path.join(root, 'log/'))
    # Model parameters
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--save_path', default=os.path.join(root, 'weights'))

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.00003, metavar='LR',
                        help='learning rate (default: 0.00003)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
    label_dims = {'UCM': 21, 'NWPU': 45, 'AID': 30, 'RSSDIVCS': 70}
    args.classnum = label_dims[args.dataset]
    if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
    return args

def get_logger(root,logname):
    """
    生成日志记录logger
    :param root:根目录
    :return: 日志记录器
    """
    logger = logging.getLogger('demo')
    logger.setLevel(level=logging.DEBUG)  # 相当于第一层过滤网

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # 输出到文件
    filename =datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_train.log'
    if not os.path.exists("./log"):
        os.mkdir("./log")
    file_handler = logging.FileHandler(os.path.join(root, './log', logname + filename))  # 设置文件名，模式，编码
    file_handler.setLevel(level=logging.INFO)  # 相当于第二层过滤网；第一层之后的内容再次过滤。
    file_handler.setFormatter(formatter)
    # 输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # 相当于第二层过滤网；第一层之后的内容再次过滤。
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def main(dataset_path):
    """
        训练过程封装
    :param dataset_path: 数据集路径
    :return:
    """
    root = os.path.dirname(os.path.abspath(__file__))
    args = get_args_parser(root, dataset_path)

    torch.manual_seed(args.seed)
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_path = os.path.abspath(dataset_path)
    # 实验时不需要加
    cutdataset.build_list(root, dataset_path, args.train_ratio)
    train_data = MyCustomDataset(os.path.join(root, 'data/train.txt'), transform=train_transform)
    test_data = MyCustomDataset(os.path.join(root, 'data/test.txt'), transform=test_transform)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_classification)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=collate_classification)

    # model = models.crossvit_base_224(num_classes=args.classnum,pretrained=True,drop_rate=0.2, attn_drop_rate=0.2,
    #              drop_path_rate=0.3)
    model = SwinVit_1.swinvit(token_dim=640, num_classes=args.classnum)
    logname = type(model).__name__ + "_" + args.dataset + "_" + str(args.train_ratio) + "_"

    logger = get_logger(root, logname)
    model.to(device)
    
    # weights_dict = torch.load("/root/autodl-tmp/MyCode/tnt_s_81.5.pth.tar", map_location='cpu')
    # weights_dict = torch.load("/root/autodl-tmp/CrossViT-main/crossvit_base_224.pth", map_location=device)['model']
  
    # 删除有关分类类别的权重
    # for k in list(weights_dict.keys()):
    #     if "head" in k:
    #         del weights_dict[k]
    # print(model.load_state_dict(weights_dict, strict=False))
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,eps=args.opt_eps)
    scheduler, _ = create_scheduler(args, optimizer)
    criteon = nn.CrossEntropyLoss(label_smoothing=0.1)
    supcon = JutinLoss.SupConLoss()
    best_acc, best_epoch = 0.0, 0
    loss_show = []
    acc_show = []
    train_acc_show = []


    for epoch in range(args.epochs):
        model.train()
        description = str(epoch) + "/" + str(args.epochs)
        right_num = 0
        train_num = len(train_loader.dataset)
        loss_runing =0
        with tqdm(train_loader, desc=description) as iterator:
            for img, label, imgs_path in iterator:
                img, label = img.to(device), label.to(device)
                logits,swin, tnt  = model(img)
                loss0 = criteon(logits, label)
                loss1 = supcon(swin, label)
                loss2 = supcon(tnt, label)
                loss = 0.6*loss0 +0.2*loss1 +0.2*loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predict_label = torch.argmax(logits, dim=1)
                right = (predict_label == label).sum().float().item()
                
                right_num += right
                train_acc = right / len(label)
                information = "train_acc:{:.4f},loss:{:.4f},lr:{:.6f}".format(train_acc, loss.item(),
                                                                       optimizer.state_dict()["param_groups"][0]["lr"])
                iterator.set_postfix_str(information)
                loss_runing +=loss.item()
                
        loss_show.append(loss_runing/train_num)
        scheduler.step(epoch)
        train_acc_show.append(right_num/train_num)

        test_acc, img_dict = evalute(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            if best_acc >0.949:
                torch.save(model.state_dict(), args.save_path+f"/{type(model).__name__}_{args.dataset}_epoch_{epoch}weight.pth")

            # with open("./result/resu", 'w') as file:
            #     for key, value in img_dict.items():
            #         file.write('{}: {}\n'.format(key, value))
        logger.info("Epoch:{}\ttest_acc:{}\ttrain_acc:{}\tloss:{}".format(epoch, test_acc,right_num/train_num, loss_runing/train_num))
        acc_show.append(test_acc)
        logger.info('BestEpoch:' + str(best_epoch)  + '   Bestacc:' + str(best_acc))

    logger.info('Best acc:{}\tBest epoch:{}'.format(best_acc,best_epoch))
    utils.show(acc_show, train_acc_show,loss_show)

def count_parameters(model):
    '''
    num of model's parameters
    :param model:
    :return: 参数数量，单位MB
    '''
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000
def evalute(model, loader):
    '''

    :param model: 模型
    :param loader: testSet loader
    :return: 准确率 预测值字典{imgpath: 预测值索引}
    '''
    model.eval()
    correct = 0.0
    total = len(loader.dataset)
    img_dict = {}

    with tqdm(loader) as iterator:
        for img, label, imgs_path in iterator:

            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                out,a,_ = model(img)
                pred = out.argmax(dim=1)
            correct += torch.eq(pred, label).sum().float().item()
            for path, pred_value in zip(imgs_path, pred):
                img_dict[path] = pred_value
    return correct / total, img_dict
