#---------------------------------------------------
# The symbiosis between SNN and ANN for EEG analysis
#---------------------------------------------------
from __future__ import print_function
from torch.utils.data.dataloader import DataLoader
import datetime
from SA_modules import *
from utils import *
from config import get_config
import time
from Feature.BuildDESEEDTool import *
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from torch.autograd import Variable
import SA_architectures
import copy


sum_k_s = 0
cnt_k_s = 0
sum_k_a = 0
cnt_k_a = 0
lam = 0.1
T = 3.0

fea_in_s = []
fea_out_s = []
def hook_s(module, input, output):
    fea_in_s.append(input)
    fea_out_s.append(output)
fea_in_a = []
fea_out_a = []
def hook_a(module, input, output):
    fea_in_a.append(input)
    fea_out_a.append(output)

def new_loss_function(target_out, out, func='KL'):
    if func == 'mse':
        target_out = target_out.detach()
        f = nn.MSELoss()
        diff_loss = f(target_out, out)
    elif func == 'cos':
        target_out = target_out.detach()
        f = nn.CosineSimilarity(dim=1, eps=1e-6)
        diff_loss = 1.0 - torch.mean(f(target_out, out))
    elif func == 'KL':
        predict = F.log_softmax(out.detach() / T, dim=1)
        target_data = F.softmax(target_out.detach() / T, dim=1)
        target = Variable(target_data.data, requires_grad=False)
        kl_loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        diff_loss = kl_loss  # * alpha
    else:
        assert False
    loss = diff_loss #+ lam * k
    return loss#, diff_loss

# ===self-training===
def indep_train():

    global best_acc_ann, best_acc_snn, max_acc_snn_test, max_acc_ann_test
    global mid_best_acc_ann, mid_best_acc_snn, mid_max_acc_snn_test, mid_max_acc_ann_test

    for epoch in range(start_epoch, args.indep_epochs + args.epochs):# + args.epochs
        losses_snn = AverageMeter()
        top1_snn = AverageMeter()
        losses_ann = AverageMeter()
        top1_ann = AverageMeter()

        model_snn.train()
        model_ann.train()

        confusion_s = ConfusionMatrix(num_classes=args.num_classes)
        confusion_a = ConfusionMatrix(num_classes=args.num_classes)

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(args.device), target.to(args.device)
            target_onehot = F.one_hot(target, args.num_classes).float()

            # ann
            output_ann = model_ann(data)
            loss_ann = F.cross_entropy(output_ann, target_onehot)

            if np.isnan(loss_ann.item()) or np.isinf(loss_ann.item()):
                print('encounter ann_loss', loss_ann)
                return False

            optimizer_ann.zero_grad()
            loss_ann.backward(retain_graph=True)
            optimizer_ann.step()

            losses_ann.update(loss_ann.item(), data.size(0))
            pred_ann = output_ann.max(1, keepdim=True)[1]
            correct_ann = pred_ann.eq(target.data.view_as(pred_ann)).cpu().sum()
            confusion_a.update(pred_ann.cpu(), target.cpu())

            functional.reset_net(model_ann)

            # snn
            optimizer_snn.zero_grad()
            output_snn = model_snn(data)
            loss_snn = F.cross_entropy(output_snn, target_onehot)

            if np.isnan(loss_snn.item()) or np.isinf(loss_snn.item()):
                print('encounter loss_snn', loss_snn)
                return False

            loss_snn.backward(retain_graph=True)
            optimizer_snn.step()

            losses_snn.update(loss_snn.item(), data.size(0))
            pred_snn = output_snn.max(1, keepdim=True)[1]
            correct_snn = pred_snn.eq(target.data.view_as(pred_snn)).cpu().sum()
            confusion_s.update(pred_snn.cpu(), target.cpu())

            functional.reset_net(model_snn)

        Accuracy_a, F1score_a = confusion_a.summary()
        Accuracy_s, F1score_s = confusion_s.summary()
        top1_ann.update(Accuracy_a)
        top1_snn.update(Accuracy_s)

        # Save checkpoint
        if epoch < args.indep_epochs:
            if top1_snn.avg > mid_best_acc_snn:
                state_snn = {
                    'net': model_snn.state_dict(),
                    'acc': top1_snn.avg,
                    'epoch': epoch
                }
                torch.save(state_snn, mid_SNN_Best)
                mid_best_acc_snn = top1_snn.avg
                best_acc_snn = mid_best_acc_snn
            if top1_ann.avg > mid_best_acc_ann:
                state_ann = {
                    'net': model_ann.state_dict(),
                    'acc': top1_ann.avg,
                    'epoch': epoch
                }
                torch.save(state_ann, mid_ANN_Best)
                mid_best_acc_ann = top1_ann.avg
                best_acc_ann = mid_best_acc_ann

            if (epoch + 1) % args.indep_epochs == 0:
                state_snn = {
                    'net': model_snn.state_dict(),
                    'acc': top1_snn.avg,
                    'epoch': epoch
                }
                torch.save(state_snn, SNN_TrainDict)
                state_ann = {
                    'net': model_ann.state_dict(),
                    'acc': top1_ann.avg,
                    'epoch': epoch
                }
                torch.save(state_ann, ANN_TrainDict)
        else:
            if top1_snn.avg > best_acc_snn:
                state_snn = {
                    'net': model_snn.state_dict(),
                    'acc': top1_snn.avg,
                    'epoch': epoch
                }
                torch.save(state_snn, SNN_Best)
                best_acc_snn = top1_snn.avg
            if top1_ann.avg > best_acc_ann:
                state_ann = {
                    'net': model_ann.state_dict(),
                    'acc': top1_ann.avg,
                    'epoch': epoch
                }
                torch.save(state_ann, ANN_Best)
                best_acc_ann = top1_ann.avg

            if (epoch + 1) % args.epochs == 0:
                state_snn = {
                    'net': model_snn.state_dict(),
                    'acc': top1_snn.avg,
                    'epoch': epoch
                }
                torch.save(state_snn, SNN_TrainDict)
                state_ann = {
                    'net': model_ann.state_dict(),
                    'acc': top1_ann.avg,
                    'epoch': epoch
                }
                torch.save(state_ann, ANN_TrainDict)

        f.write('\n indep_Epoch: {}, lr: {:.1e}, loss_snn: {:.4f}, acc_snn: {:.4f}, loss_ann: {:.4f}, acc_ann: {:.4f}'
                        .format(epoch, args.lr, losses_snn.avg, top1_snn.avg, losses_ann.avg, top1_ann.avg))

        if epoch < args.indep_epochs:
            snn_test, ann_test = test(epoch, mid_max_acc_snn_test, mid_max_acc_ann_test)
            mid_max_acc_snn_test = snn_test
            mid_max_acc_ann_test = ann_test
            max_acc_snn_test = mid_max_acc_snn_test
            max_acc_ann_test = mid_max_acc_ann_test
        else:
            snn_test, ann_test = test(epoch, max_acc_snn_test, max_acc_ann_test)
            max_acc_snn_test = snn_test
            max_acc_ann_test = ann_test

        f.write('\n mid_test__best_snn: {:.4f}, best_ann: {:.4f}'
                '\n in_test__best_snn: {:.4f}, best_ann: {:.4f}'
                        .format(mid_max_acc_snn_test, mid_max_acc_ann_test,
                                max_acc_snn_test, max_acc_ann_test))

# ===self-training===

# ===co-training===
def co_train_search():

    global sum_k_s, cnt_k_s, sum_k_a, cnt_k_a, fea_out_a, fea_in_a, fea_in_s, fea_out_s
    global mid_best_acc_ann, mid_best_acc_snn, mid_max_acc_snn_test, mid_max_acc_ann_test
    global co_best_acc_ann, co_best_acc_snn, co_max_acc_snn_test, co_max_acc_ann_test, top_k_acc_snn, top_k_acc_ann, final_acc_snn, final_acc_ann

    co_best_acc_ann = mid_best_acc_ann
    co_best_acc_snn = mid_best_acc_snn
    co_max_acc_snn_test = mid_max_acc_snn_test
    co_max_acc_ann_test = mid_max_acc_ann_test
    top_k_acc_snn = mid_max_acc_snn_test
    top_k_acc_ann = mid_max_acc_ann_test
    final_acc_snn = co_max_acc_snn_test
    final_acc_ann = co_max_acc_ann_test

    # if os.path.exists(mid_SNN_Best):
    #     model_snn.load_state_dict(torch.load(mid_SNN_Best)['net'])
    # if os.path.exists(mid_ANN_Best):
    #     model_ann.load_state_dict(torch.load(mid_ANN_Best)['net'])

    for epoch in range(start_epoch, args.epochs):

        losses_snn = AverageMeter()
        top1_snn = AverageMeter()
        losses_ann = AverageMeter()
        top1_ann = AverageMeter()

        model_snn.train()
        model_ann.train()
        start_time = time.time()

        confusion_s = ConfusionMatrix(num_classes=args.num_classes)
        confusion_a = ConfusionMatrix(num_classes=args.num_classes)

        for batch_idx, (data, target) in enumerate(train_loader):
            sum_k_s = 0
            cnt_k_s = 0
            sum_k_a = 0
            cnt_k_a = 0
            data, target = data.float().to(args.device), target.to(args.device)
            target_onehot = F.one_hot(target, args.num_classes).float()

            # ann
            output_ann = model_ann(data).to(args.device)
            output_snn = model_snn(data).to(args.device)

            optimizer_ann.zero_grad()
            loss_ann = F.cross_entropy(output_ann, target_onehot) #+ new_loss_function(output_snn, output_ann, func='KL')

            if np.isnan(loss_ann.item()) or np.isinf(loss_ann.item()):
                print('encounter ann_loss', loss_ann)
                return False

            loss_ann.backward(retain_graph=True)
            optimizer_ann.step()

            losses_ann.update(loss_ann.item(), data.size(0))
            pred_ann = output_ann.max(1, keepdim=True)[1]
            correct_ann = pred_ann.eq(target.data.view_as(pred_ann)).cpu().sum()
            confusion_a.update(pred_ann.cpu(), target.cpu())
            functional.reset_net(model_ann)

            # snn
            optimizer_snn.zero_grad()
            loss_snn = F.cross_entropy(output_snn, target_onehot) #+ new_loss_function(output_ann, output_snn, func='KL')

            if np.isnan(loss_snn.item()) or np.isinf(loss_snn.item()):
                print('encounter loss_snn', loss_snn)
                return False

            loss_snn.backward(retain_graph=True)
            optimizer_snn.step()

            losses_snn.update(loss_snn.item(), data.size(0))
            pred_snn = output_snn.max(1, keepdim=True)[1]
            correct_snn = pred_snn.eq(target.data.view_as(pred_snn)).cpu().sum()
            confusion_s.update(pred_snn.cpu(), target.cpu())
            functional.reset_net(model_snn)

        Accuracy_a, F1score_a = confusion_a.summary()
        Accuracy_s, F1score_s = confusion_s.summary()
        top1_ann.update(Accuracy_a)
        top1_snn.update(Accuracy_s)

        if top1_ann.avg > co_best_acc_ann:
            co_best_acc_ann = top1_ann.avg
        if top1_snn.avg > co_best_acc_snn:
            co_best_acc_snn = top1_snn.avg

        f.write('\n Co_Epoch: {}, target: {}, lr: {:.1e}, loss_snn: {:.4f}, acc_snn: {:.4f}, loss_ann: {:.4f}, acc_ann: {:.4f}'
                        .format(epoch, target_sample, args.lr, losses_snn.avg, top1_snn.avg, losses_ann.avg, top1_ann.avg))
        co_snn_test, co_ann_test = test(epoch, co_max_acc_snn_test, co_max_acc_ann_test)
        co_max_acc_snn_test = co_snn_test
        co_max_acc_ann_test = co_ann_test
        final_acc_snn = co_max_acc_snn_test
        final_acc_ann = co_max_acc_ann_test

        f.write('\n in_test__best_snn: {:.4f}, best_ann: {:.4f}'
                '\n co_test__best_snn: {:.4f}, best_ann: {:.4f}'
                        .format(max_acc_snn_test, max_acc_ann_test,
                                co_max_acc_snn_test, co_max_acc_ann_test))

        if (epoch+1) % args.GA_iter == 0 or (epoch+1) == args.epochs:

            torch.manual_seed(args.search_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(args.search_seed)

            if os.path.exists(SNN_Test_Best):
                model_snn.load_state_dict(torch.load(SNN_Test_Best)['net'])
            if os.path.exists(ANN_Test_Best):
                model_ann.load_state_dict(torch.load(ANN_Test_Best)['net'])

            worker = evolution_algo.EvolutionarySearch(args, model_snn, model_ann)
            history_s, history_a = worker.search(args.GA_epoch, test_loader, train_loader)

            # from last top-k, return top-k
            return_s_top_k = history_s[-1][:args.return_topk][0] # [0]: net, [1]: acc
            return_a_top_k = history_a[-1][:args.return_topk][0]
            # save the best history
            if return_s_top_k[1] > top_k_acc_snn:
                top_k_acc_snn = return_s_top_k[1]
                torch.save(return_s_top_k, SNN_Best_Arch_GA)
            if return_a_top_k[1] > top_k_acc_ann:
                top_k_acc_ann = return_a_top_k[1]
                torch.save(return_a_top_k, ANN_Best_Arch_GA)

            if os.path.exists(SNN_Best_Arch_GA):
                model_snn.load_state_dict(torch.load(SNN_Best_Arch_GA)[0])
            if os.path.exists(ANN_Best_Arch_GA):
                model_ann.load_state_dict(torch.load(ANN_Best_Arch_GA)[0])

        if top_k_acc_snn > co_max_acc_snn_test:
            final_acc_snn = top_k_acc_snn
        if top_k_acc_ann > co_max_acc_ann_test:
            final_acc_ann = top_k_acc_ann

        f.write('\n in_test__best_snn: {:.4f}, best_ann: {:.4f}'
                '\n co_test__best_snn: {:.4f}, best_ann: {:.4f}'
                '\n co_GA__top_k_snn: {:.4f}, top_k_ann: {:.4f}'
                '\n final__best_snn: {:.4f}, best_ann: {:.4f}'
                        .format(max_acc_snn_test, max_acc_ann_test,
                                co_max_acc_snn_test, co_max_acc_ann_test,
                                top_k_acc_snn, top_k_acc_ann,
                                final_acc_snn, final_acc_ann))



def test(epoch, acc_snn_test, acc_ann_test):

    losses_snn = AverageMeter()
    top1_snn = AverageMeter()
    losses_ann = AverageMeter()
    top1_ann = AverageMeter()

    confusion_s = ConfusionMatrix(num_classes=args.num_classes)
    confusion_a = ConfusionMatrix(num_classes=args.num_classes)

    with torch.no_grad():
        model_snn.eval()
        model_ann.eval()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.float().to(args.device), target.to(args.device)
            target_onehot = F.one_hot(target, args.num_classes).float()

            # snn
            output_snn = model_snn(data).to(args.device)
            loss_snn = F.cross_entropy(output_snn, target_onehot)

            losses_snn.update(loss_snn.item(), data.size(0))
            pred_snn = output_snn.max(1, keepdim=True)[1]
            correct_snn = pred_snn.eq(target.data.view_as(pred_snn)).cpu().sum()
            confusion_s.update(pred_snn.cpu(), target.cpu())

            # ann
            output_ann = model_ann(data).to(args.device)
            loss_ann = F.cross_entropy(output_ann, target)

            losses_ann.update(loss_ann.item(), data.size(0))
            pred_ann = output_ann.max(1, keepdim=True)[1]
            correct_ann = pred_ann.eq(target.data.view_as(pred_ann)).cpu().sum()
            confusion_a.update(pred_ann.cpu(), target.cpu())

        Accuracy_a, F1score_a = confusion_a.summary()
        Accuracy_s, F1score_s = confusion_s.summary()
        top1_ann.update(Accuracy_a)
        top1_snn.update(Accuracy_s)

        if top1_snn.avg > acc_snn_test:
            acc_snn_test = top1_snn.avg
            state_snn = {
                    'acc': max_acc_snn_test,
                    'epoch': epoch,
                    'net': model_snn.state_dict()
                }
            torch.save(state_snn, SNN_Test_Best)

        if top1_ann.avg > acc_ann_test:
            acc_ann_test = top1_ann.avg
            state_ann = {
                    'acc': max_acc_ann_test,
                    'epoch': epoch,
                    'net': model_ann.state_dict()
                }
            torch.save(state_ann, ANN_Test_Best)

        f.write('\n test_loss_snn: {:.4f}, test_acc_snn: {:.4f}, test_loss_ann: {:.4f}, test_acc_ann: {:.4f}'
            .format(losses_snn.avg, top1_snn.avg, losses_ann.avg, top1_ann.avg)
        )
    return acc_snn_test, acc_ann_test

if __name__ == '__main__':

    args = get_config()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # ==divide
    datasetList = ['SEED']  # 'Fatigue', 'SEED', 'Cognitive'

    # ==Dataset
    for d, dataset in enumerate(datasetList):
        args.dataset = dataset
        print('dataset:', args.dataset)

        # save print
        path = os.path.abspath(os.path.dirname(__file__)) + '/logs/Prints/' + args.dataset
        timeShow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        sys.stdout = Logger(filename=path + '/GA_SANN_cross_' + timeShow + args.dataset + '_Print.txt')

        if args.dataset == 'Fatigue':
            # data load, one subject used to test
            sub_list = ['ARCALE', 'ANZALE', 'BORGIA', 'CULLEO', 'CILRAM', 'CALGIO', 'DESTER', 'DIFANT', 'GNATN', 'MESMAR', 'MARFRA', 'SCAEMI', 'SALSTE', 'VALNIC', 'VALPAO']
            data_dir_list = ['./data_set/Fatigue/']
            sampleNum = 1400
            num_feats = 1647
            num_nodes = 61  # channels
            num_freq = 27
            args.input_channels = 5
            args.num_classes = 2 # classes of label
            args.channels_FC1 = 4
            args.channels_FC2 = 5
            args.input_shape1 = 8
            args.input_shape2 = 9
            args.channels_ANN1 = 4
            args.channels_ANN2 = 4
            args.channels_alexnet1 = 5
            args.channels_alexnet2 = 6

        elif args.dataset == 'SEED':
            sub_list = ['1', '2', '3', '4', '5', '6', '7', '8',
                        '9', '10', '11', '12', '13', '14', '15']
            data_dir_list = ['./data_set/SEED1/', './data_set/SEED2/', './data_set/SEED3/']
            sampleNum = 842
            num_feats = 2604
            num_nodes = 62
            num_freq = 42
            args.input_channels = 5
            args.num_classes = 3  # classes of label
            args.channels_FC1 = 4
            args.channels_FC2 = 5
            args.input_shape1 = 8
            args.input_shape2 = 9
            args.channels_ANN1 = 4
            args.channels_ANN2 = 4
            args.channels_alexnet1 = 5
            args.channels_alexnet2 = 6

        elif args.dataset == 'Cognitive':

            sub_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            data_dir_list = ['./data_set/Cog_happiness/', './data_set/Cog_neutral/', './data_set/Cog_sadness/']
            num_nodes = 32
            # num_freq = 0.5-30Hz
            args.input_channels = 4
            args.num_classes = 3 # classes of label
            args.channels_FC1 = 3
            args.channels_FC2 = 1
            args.input_shape1 = 7
            args.input_shape2 = 5
            args.channels_ANN1 = 3
            args.channels_ANN2 = 2
            args.channels_alexnet1 = 4
            args.channels_alexnet2 = 2

        else:
            print('error!')
            os._exit(0)

        # print: all session
        avg_All_snn = AverageMeter()  # all datasets avg acc
        avg_All_ann = AverageMeter()  # all datasets avg acc
        mid_avg_All_snn = AverageMeter()  # all datasets avg acc
        mid_avg_All_ann = AverageMeter()  # all datasets avg acc
        co_avg_All_snn = AverageMeter()  # all datasets avg acc
        co_avg_All_ann = AverageMeter()
        topk_avg_All_snn = AverageMeter()
        topk_avg_All_ann = AverageMeter()
        final_avg_All_snn = AverageMeter()
        final_avg_All_ann = AverageMeter()
        avg_All_snn_array = []
        avg_All_ann_array = []
        mid_avg_All_snn_array = []
        mid_avg_All_ann_array = []
        co_avg_All_snn_array = []
        co_avg_All_ann_array = []
        topk_avg_All_snn_array = []
        topk_avg_All_ann_array = []
        final_avg_All_snn_array = []
        final_avg_All_ann_array = []

        # ==Session
        for k in range(len(data_dir_list)):
            folder_path = data_dir_list[k]
            print('======folder_path:', folder_path)
            Num_target = 0  # number of target
            # Raw data .mat files processed into npy files
            origin_path = 'E:/Projects_Code/22.5-SNN/Datasets-Origin/SEED-DE/' + str(k + 1) + '/'

            feature_vector_dict, label_dict = build_DE_eeg_dataset(folder_path, origin_path, dis=6, map_size=9)

            # print: all subjects
            best_acc_array_snn = []
            avg_best_acc_snn = AverageMeter()
            best_acc_array_ann = []
            avg_best_acc_ann = AverageMeter()
            mid_best_acc_array_snn = []
            mid_avg_best_acc_snn = AverageMeter()
            mid_best_acc_array_ann = []
            mid_avg_best_acc_ann = AverageMeter()
            co_best_acc_array_snn = []
            co_avg_best_acc_snn = AverageMeter()
            co_best_acc_array_ann = []
            co_avg_best_acc_ann = AverageMeter()
            top_k_acc_snn_array = []
            top_k_acc_snn_avg = AverageMeter()
            top_k_acc_ann_array = []
            top_k_acc_ann_avg = AverageMeter()
            final_acc_snn_array = []
            final_acc_snn_avg = AverageMeter()
            final_acc_ann_array = []
            final_acc_ann_avg = AverageMeter()

            # ==Leave-one-subject-out LOSO
            for m, target_sample in enumerate(sub_list):
                Num_target = Num_target + 1
                test_subjects = [target_sample]
                print('target_sample: ', test_subjects)

                info = [args.dataset, target_sample]
                save_name = '_'.join(info)
                log_dir = './Trained_models/SANN_GA_cross/' + save_name + '_cross'
                if not os.path.isdir(log_dir):
                    os.makedirs(log_dir)
                SNN_Best = log_dir + '/snn_best.pth'  # the best para of indep_train
                ANN_Best = log_dir + '/ann_best.pth'
                mid_SNN_Best = log_dir + '/snn_best_mid.pth'  # the best para of mid_train
                mid_ANN_Best = log_dir + '/ann_best_mid.pth'
                SNN_TrainDict = log_dir + '/snn_pt_scheduled.pth'  # the final para of indep_train
                ANN_TrainDict = log_dir + '/ann_pt_scheduled.pth'
                SNN_Best_Arch_GA = log_dir + '/snn_best_arch_GA.pth'  # the best arch of GA
                ANN_Best_Arch_GA = log_dir + '/ann_best_arch_GA.pth'
                SNN_Test_Best = log_dir + '/snn_test_best.pth'  # the best acc of test
                ANN_Test_Best = log_dir + '/ann_test_best.pth'

                train_feature, train_label, test_feature, test_label = subject_cross_data_split(feature_vector_dict, label_dict, test_subjects, args.dataset)
                train_set = EEGDataset_DE(train_feature, train_label)
                test_set = EEGDataset_DE(test_feature, test_label)

                train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=args.b,
                    shuffle=True,
                    num_workers=args.j,
                    drop_last=True,
                    pin_memory=True)

                test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=args.b,
                    shuffle=False,
                    num_workers=args.j,
                    drop_last=True,
                    pin_memory=True)

                # SA_architectures
                model = SA_architectures.__dict__[args.arch](num_classes=args.num_classes, dropout=0, in_channels=args.input_channels, args=args)
                model = replace_maxpool2d_by_avgpool2d(model)

                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if hasattr(m, 'bias') and m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, val=1)
                        nn.init.zeros_(m.bias)

                model_ann = copy.deepcopy(model)
                model_ann.to(args.device)

                model_snn = replace_relu_by_spikingnorm(model, True)
                model_snn.to(args.device)

                # Training settings
                optimizer_snn = None
                optimizer_ann = None
                if args.opt == 'SGD':
                    optimizer_snn = torch.optim.SGD(model_snn.parameters(), lr=args.lr, momentum=args.momentum)
                    optimizer_ann = torch.optim.SGD(model_ann.parameters(), lr=args.lr, momentum=args.momentum)
                elif args.opt == 'Adam':
                    optimizer_snn = torch.optim.Adam(model_snn.parameters(), lr=args.lr)
                    optimizer_ann = torch.optim.Adam(model_ann.parameters(), lr=args.lr)
                else:
                    raise NotImplementedError(args.opt)

                lr_scheduler_snn = None
                lr_scheduler_ann = None
                if args.lr_scheduler == 'StepLR':
                    lr_scheduler_snn = torch.optim.lr_scheduler.StepLR(optimizer_snn, step_size=args.step_size, gamma=args.gamma)
                    lr_scheduler_ann = torch.optim.lr_scheduler.StepLR(optimizer_ann, step_size=args.step_size, gamma=args.gamma)
                elif args.lr_scheduler == 'CosALR':
                    lr_scheduler_snn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_snn, T_max=args.T_max)
                    lr_scheduler_ann = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ann, T_max=args.T_max)
                else:
                    raise NotImplementedError(args.lr_scheduler)

                log_file = log_dir + '/logs.log'

                if args.log:
                    f = open(log_file, 'w', buffering=1)
                else:
                    f = sys.stdout

                f.write('\n Run on time: {}'.format(datetime.datetime.now()))

                f.write('\n snn: {}'.format(model_snn))
                f.write('\n ann: {}'.format(model_ann))

                # print: each subject
                max_acc_snn_test = 0
                max_acc_ann_test = 0
                best_acc_ann = 0
                best_acc_snn = 0
                mid_max_acc_snn_test = 0
                mid_max_acc_ann_test = 0
                mid_best_acc_ann = 0
                mid_best_acc_snn = 0
                co_max_acc_snn_test = 0
                co_max_acc_ann_test = 0
                co_best_acc_ann = 0
                co_best_acc_snn = 0
                top_k_acc_snn = 0
                top_k_acc_ann = 0
                final_acc_snn = 0
                final_acc_ann = 0
                start_epoch = 0

                # ======self-training======
                print(f'\n ---indep---tag-sam={args.dataset}{k+1}')
                indep_train()
                # ======self-training======

                # ======Co-Training======
                print(f'\n ---coTrain---tag-sam={args.dataset}{k+1}')
                co_train_search()
                # ======Co-Training======

                f.write('\n mid_Test epochs: {}, SNN Highest accuracy: {:.4f}, ANN Highest accuracy: {:.4f}'.format(args.indep_epochs, mid_max_acc_snn_test, mid_max_acc_ann_test))
                f.write('\n Test total_epochs: {}, SNN Highest accuracy: {:.4f}, ANN Highest accuracy: {:.4f}'.format(args.epochs + args.indep_epochs, max_acc_snn_test, max_acc_ann_test))
                f.write('\n co_Test epochs: {}, SNN Highest accuracy: {:.4f}, ANN Highest accuracy: {:.4f}'.format(args.epochs, co_max_acc_snn_test, co_max_acc_ann_test))
                f.write('\n co_GA_Test epochs: {}, SNN Highest accuracy: {:.4f}, ANN Highest accuracy: {:.4f}'.format(args.GA_epoch, top_k_acc_snn, top_k_acc_ann))
                f.write('\n Final_Test epochs: {}, SNN Highest accuracy: {:.4f}, ANN Highest accuracy: {:.4f}'.format(args.epochs, final_acc_snn, final_acc_ann))

                mid_best_acc_array_snn.append(mid_max_acc_snn_test)
                mid_avg_best_acc_snn.update(mid_max_acc_snn_test)
                mid_best_acc_array_ann.append(mid_max_acc_ann_test)
                mid_avg_best_acc_ann.update(mid_max_acc_ann_test)
                print(f'\n mid_Sessions: dataset={folder_path}, target_sample={target_sample}, '
                      f'\n mid_avg_best_acc_snn={mid_avg_best_acc_snn.avg}, mid_best_arr_snn={mid_best_acc_array_snn}, '
                      f'\n mid_avg_best_acc_ann={mid_avg_best_acc_ann.avg}, mid_best_arr_ann={mid_best_acc_array_ann}')

                best_acc_array_snn.append(max_acc_snn_test)
                avg_best_acc_snn.update(max_acc_snn_test)
                best_acc_array_ann.append(max_acc_ann_test)
                avg_best_acc_ann.update(max_acc_ann_test)
                print(f'\n Sessions: dataset={folder_path}, target_sample={target_sample}, '
                      f'\n avg_best_acc_snn={avg_best_acc_snn.avg}, best_arr_snn={best_acc_array_snn}, '
                      f'\n avg_best_acc_ann={avg_best_acc_ann.avg}, best_arr_ann={best_acc_array_ann}')

                co_best_acc_array_snn.append(co_max_acc_snn_test)
                co_avg_best_acc_snn.update(co_max_acc_snn_test)
                co_best_acc_array_ann.append(co_max_acc_ann_test)
                co_avg_best_acc_ann.update(co_max_acc_ann_test)
                top_k_acc_snn_array.append(top_k_acc_snn)
                top_k_acc_snn_avg.update(top_k_acc_snn)
                top_k_acc_ann_array.append(top_k_acc_ann)
                top_k_acc_ann_avg.update(top_k_acc_ann)
                final_acc_snn_array.append(final_acc_snn)
                final_acc_snn_avg.update(final_acc_snn)
                final_acc_ann_array.append(final_acc_ann)
                final_acc_ann_avg.update(final_acc_ann)
                print(f'\n co_Sessions: dataset={folder_path}, target_sample={target_sample}, '
                      f'\n avg_best_acc_snn={co_avg_best_acc_snn.avg}, best_arr_snn={co_best_acc_array_snn}, '
                      f'\n avg_best_acc_ann={co_avg_best_acc_ann.avg}, best_arr_ann={co_best_acc_array_ann}, '
                      f'\n avg_top_k_snn={top_k_acc_snn_avg.avg}, top_k_arr_snn={top_k_acc_snn_array}, '
                      f'\n avg_top_k_ann={top_k_acc_ann_avg.avg}, top_k_arr_ann={top_k_acc_ann_array}, '
                      f'\n avg_final_snn={final_acc_snn_avg.avg}, final_arr_snn={final_acc_snn_array}, '
                      f'\n avg_final_ann={final_acc_ann_avg.avg}, final_arr_ann={final_acc_ann_array}')

            mid_avg_All_snn_array.append(mid_avg_best_acc_snn.avg)
            print(f'\n -----Array-----'
                  f'\n mid_dataset: mid_avg_all_snn_array={mid_avg_All_snn_array}')
            mid_avg_All_ann_array.append(mid_avg_best_acc_ann.avg)
            print(f'mid_dataset: mid_avg_all_ann_array={mid_avg_All_ann_array}')

            avg_All_snn_array.append(avg_best_acc_snn.avg)
            print(f'dataset: avg_all_snn_array={avg_All_snn_array}')
            avg_All_ann_array.append(avg_best_acc_ann.avg)
            print(f'dataset: avg_all_ann_array={avg_All_ann_array}')

            co_avg_All_snn_array.append(co_avg_best_acc_snn.avg)
            print(f'Co_dataset: co_avg_all_snn_array={co_avg_All_snn_array}')
            co_avg_All_ann_array.append(co_avg_best_acc_ann.avg)
            print(f'Co_dataset: co_avg_all_ann_array={co_avg_All_ann_array}')

            topk_avg_All_snn_array.append(top_k_acc_snn_avg.avg)
            print(f'Topk_dataset: topk_avg_All_snn_array={topk_avg_All_snn_array}')
            topk_avg_All_ann_array.append(top_k_acc_ann_avg.avg)
            print(f'Topk_dataset: topk_avg_All_ann_array={topk_avg_All_ann_array}')

            final_avg_All_snn_array.append(final_acc_snn_avg.avg)
            print(f'final_dataset: final_avg_All_snn_array={final_avg_All_snn_array}')
            final_avg_All_ann_array.append(final_acc_ann_avg.avg)
            print(f'final_dataset: final_avg_All_ann_array={final_avg_All_ann_array}')

            # avg of final
            mid_avg_All_snn.update(mid_avg_best_acc_snn.avg)
            print(f'\n dataset={args.dataset}'
                  f'\n mid_snn={mid_avg_All_snn.avg}')
            mid_avg_All_ann.update(mid_avg_best_acc_ann.avg)
            print(f'mid_ann={mid_avg_All_ann.avg}')

            avg_All_snn.update(avg_best_acc_snn.avg)
            print(f'indep_snn={avg_All_snn.avg}')
            avg_All_ann.update(avg_best_acc_ann.avg)
            print(f'indep_ann={avg_All_ann.avg}')

            co_avg_All_snn.update(co_avg_best_acc_snn.avg)
            print(f'co_snn={co_avg_All_snn.avg}')
            co_avg_All_ann.update(co_avg_best_acc_ann.avg)
            print(f'co_ann={co_avg_All_ann.avg}')

            topk_avg_All_snn.update(top_k_acc_snn_avg.avg)
            print(f'topk_snn={topk_avg_All_snn.avg}')
            topk_avg_All_ann.update(top_k_acc_ann_avg.avg)
            print(f'topk_ann={topk_avg_All_ann.avg}')

            final_avg_All_snn.update(final_acc_snn_avg.avg)
            print(f'final_snn={final_avg_All_snn.avg}')
            final_avg_All_ann.update(final_acc_ann_avg.avg)
            print(f'final_ann={final_avg_All_ann.avg}')
