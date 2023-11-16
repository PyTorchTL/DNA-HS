#---------------------------------------------------
# The symbiosis between SNN and ANN for EEG analysis
#---------------------------------------------------

import torch
import numpy as np
import logging
from copy import deepcopy
from spikingjelly.clock_driven import functional
from SANN_Index import *
import torch.nn.functional as F
import torch.nn as nn
from torcheeg.models import DGCNN

class EvolutionarySearch(object):

    def __init__(self, args, model_snn, model_ann):
        super(EvolutionarySearch, self).__init__()

        self.args = args
        self.model_snn = model_snn
        self.model_ann = model_ann

        self.num_mutation = args.num_mutation
        self.mutation_prob = args.mutation_prob
        self.num_crossover = args.num_crossover
        
        self.num_topk = args.num_topk
        self.history_topk_s = [] # including top-k candidates at every iteration during the search
        self.history_topk_a = []

        self.seen_arch_s = []
        self.seen_arch_a = []

    def _random_pool(self, pool, pool_size):
        new_pool = []
        while len(new_pool) < pool_size:
            cand_arch = self.net._uniform_sampling().tolist() # block_ids
            if (cand_arch not in new_pool) and (cand_arch not in self.seen_archs):
                new_pool.append(cand_arch)

        return new_pool

    def _mutation(self, cand_s, cand_a, pool_s, pool_a, acc_s, acc_a, pool_size, test_loader, train_loader, rate):
        new_pool_s = []
        new_pool_a = []
        new_fitness_s = []
        new_fitness_a = []
        new_acc_s = []
        new_acc_a = []
        new_F_s = []
        new_F_a = []
        new_R_s = []
        new_R_a = []
        new_Pre_s = []
        new_Pre_a = []
        CM_s = []
        CM_a = []
        score_s = []
        score_a = []
        label_s = []
        label_a = []
        # variable rate
        self.mutation_prob = rate  # [0, 0.5]
        for _ in range(pool_size): # time limit
            sel_s = np.random.choice(range(len(pool_s)), 1)[0]
            sel_a = np.random.choice(range(len(pool_a)), 1)[0]
            cand_s.load_state_dict(pool_s[sel_s])
            cand_a.load_state_dict(pool_a[sel_a])

            if acc_s[sel_s] > acc_a[sel_a]:
                self.args.scalePara = 0.3  # for weighting para_weight
            else:
                self.args.scalePara = 0.7

            num_mut = 0
            # mutation: Genetic mutations that introduce additional model partial parameters
            pre_s_weight = None
            for (s, a) in zip(cand_s.modules(), cand_a.modules()):
                if isinstance(s, (nn.Linear, nn.Conv2d, DGCNN)) and isinstance(a, (nn.Conv2d, nn.Linear, DGCNN)):
                    if np.random.random() < self.mutation_prob:
                        if hasattr(s, 'weight') and hasattr(a, 'weight'):
                            pre_s_weight = s.weight.data
                            s.weight.data = s.weight.data * (1 - self.args.scalePara) + a.weight.data * self.args.scalePara
                            a.weight.data = a.weight.data * self.args.scalePara + pre_s_weight * (1 - self.args.scalePara)
                        num_mut += 1

            mut_state_s = self.infer(test_loader, train_loader, cand_s, self.args)
            mut_acc_s_v = mut_state_s['acc_test']
            new_acc_s.append(mut_state_s['acc_test'])
            new_F_s.append(mut_state_s['F_test'])
            new_R_s.append(mut_state_s['R_test'])
            new_Pre_s.append(mut_state_s['Pre_test'])
            mut_fitness_s_v = mut_state_s['acc_sum']
            new_fitness_s.append(mut_state_s['acc_sum'])
            new_pool_s.append(mut_state_s['net'])
            CM_s.append(mut_state_s['CM'])
            score_s.append(mut_state_s['score'])
            label_s.append(mut_state_s['label'])

            mut_state_a = self.infer(test_loader, train_loader, cand_a, self.args)
            mut_acc_a_v = mut_state_a['acc_test']
            new_acc_a.append(mut_state_a['acc_test'])
            new_F_a.append(mut_state_a['F_test'])
            new_R_a.append(mut_state_a['R_test'])
            new_Pre_a.append(mut_state_a['Pre_test'])
            mut_fitness_a_v = mut_state_a['acc_sum']
            new_fitness_a.append(mut_state_a['acc_sum'])
            new_pool_a.append(mut_state_a['net'])
            CM_a.append(mut_state_a['CM'])
            score_a.append(mut_state_a['score'])
            label_a.append(mut_state_a['label'])

            print(f'rate: {rate:.4f}, mut_acc_s: {mut_acc_s_v:.4f}, mut_acc_a: {mut_acc_a_v:.4f}, mut_fitness_s: {mut_fitness_s_v:.4f}, mut_fitness_a: {mut_fitness_a_v:.4f}')

        return new_pool_s, new_pool_a, new_acc_s, new_acc_a, new_F_s, new_F_a, new_R_s, new_R_a, new_Pre_s, new_Pre_a, new_fitness_s, new_fitness_a, CM_s, CM_a, score_s, score_a, label_s, label_a

    def _crossover(self, cro_cand_s, cro_cand_a, pool_s, pool_a, acc_s, acc_a, pool_size, test_loader, train_loader, rate):
        cro_pool_s = []
        cro_pool_a = []
        cro_fitness_s = []
        cro_fitness_a = []
        cro_acc_s = []
        cro_acc_a = []
        cro_F_s = []
        cro_F_a = []
        cro_R_s = []
        cro_R_a = []
        cro_Pre_s = []
        cro_Pre_a = []
        CM_s = []
        CM_a = []
        score_s = []
        score_a = []
        label_s = []
        label_a = []
        for _ in range(pool_size): # time limit
            # select cross models
            parent_1 = np.random.choice(range(len(pool_s)), 1, replace=False)
            parent_2 = np.random.choice(range(len(pool_a)), 1, replace=False)
            parent_1 = pool_s[parent_1[0]]
            parent_2 = pool_a[parent_2[0]]
            # load models para
            cro_cand_s.load_state_dict(parent_1)
            cro_cand_a.load_state_dict(parent_2)
            # cross
            l_num = 0
            l_index_s = 0
            l_index_a = 0
            for s in cro_cand_s.modules():
                if isinstance(s, (nn.Linear, nn.Conv2d, DGCNN)):
                    l_num += 1
            # variable rate
            division_pos = np.random.choice(range(1, l_num), max(1, int(np.floor(l_num * rate))))
            # division_pos = np.random.choice(range(1, l_num), 1)
            for pos, (s, a) in enumerate(zip(cro_cand_s.modules(), cro_cand_a.modules())):
                if isinstance(s, (nn.Linear, nn.Conv2d, DGCNN)):
                    l_index_s += 1
                    if l_index_s in division_pos:
                    # if l_index_s < division_pos:   ## ablation: crossPose
                        if hasattr(s, 'weight') and hasattr(a, 'weight'):
                            pre_s_weight = s.weight.data
                            s.weight.data = a.weight.data
                if isinstance(a, (nn.Conv2d, nn.Linear, DGCNN)):
                    l_index_a += 1
                    if l_index_a in division_pos:
                    # if l_index_a < division_pos:   ## ablation: crossPose
                        if hasattr(a, 'weight') and hasattr(s, 'weight'):
                            a.weight.data = pre_s_weight

            cro_state_s = self.infer(test_loader, train_loader, cro_cand_s, self.args)
            cro_acc_s_value = cro_state_s['acc_test']
            cro_acc_s.append(cro_state_s['acc_test'])
            cro_F_s.append(cro_state_s['F_test'])
            cro_R_s.append(cro_state_s['R_test'])
            cro_Pre_s.append(cro_state_s['Pre_test'])
            cro_fitness_s_value = cro_state_s['acc_sum']
            cro_fitness_s.append(cro_state_s['acc_sum'])
            cro_pool_s.append(cro_state_s['net'])
            CM_s.append(cro_state_s['CM'])
            score_s.append(cro_state_s['score'])
            label_s.append(cro_state_s['label'])

            cro_state_a = self.infer(test_loader, train_loader, cro_cand_a, self.args)
            cro_acc_a_value = cro_state_a['acc_test']
            cro_acc_a.append(cro_state_a['acc_test'])
            cro_F_a.append(cro_state_a['F_test'])
            cro_R_a.append(cro_state_a['R_test'])
            cro_Pre_a.append(cro_state_a['Pre_test'])
            cro_fitness_a_value = cro_state_a['acc_sum']
            cro_fitness_a.append(cro_state_a['acc_sum'])
            cro_pool_a.append(cro_state_a['net'])
            CM_a.append(cro_state_a['CM'])
            score_a.append(cro_state_a['score'])
            label_a.append(cro_state_a['label'])

            print(f'rate: {rate:.4f}, cro_acc_s: {cro_acc_s_value:.4f}, cro_acc_a: {cro_acc_a_value:.4f}, cro_fitness_s: {cro_fitness_s_value:.4f}, cro_fitness_a: {cro_fitness_a_value:.4f}')

        return cro_pool_s, cro_pool_a, cro_acc_s, cro_acc_a, cro_F_s, cro_F_a, cro_R_s, cro_R_a, cro_Pre_s, cro_Pre_a, cro_fitness_s, cro_fitness_a, CM_s, CM_a, score_s, score_a, label_s, label_a


    def search(self, max_search_iter, test_loader, train_loader):

        topk_pool_s = []
        topk_pool_a = []
        topk_acc_s = []
        topk_acc_a = []
        topk_F_s = []
        topk_F_a = []
        topk_R_s = []
        topk_R_a = []
        topk_Pre_s = []
        topk_Pre_a = []
        topk_fitness_s = []
        topk_fitness_a = []
        topk_CM_s = []
        topk_CM_a = []
        topk_score_s = []
        topk_score_a = []
        topk_label_s = []
        topk_label_a = []
        cand_s = deepcopy(self.model_snn)
        cand_a = deepcopy(self.model_ann)
        for it in range(max_search_iter):
            print(f'search_iter: {it}')
            acc_s_list = []
            acc_a_list = []
            F_s_list = []
            F_a_list = []
            R_s_list = []
            R_a_list = []
            Pre_s_list = []
            Pre_a_list = []
            fitness_s_list = []
            fitness_a_list = []
            topk_s_list = []
            topk_a_list = []
            avg_topk_s = AverageMeter()
            avg_topk_a = AverageMeter()
            avg_topk_fitness_s = AverageMeter()
            avg_topk_fitness_a = AverageMeter()
            CM_s = []
            CM_a = []
            score_s = []
            score_a = []
            label_s = []
            label_a = []
            # variable rate
            rate = 1.0 - (1/2) ** (1.0 - it / max_search_iter)
            if it == 0:
                state_s = self.infer(test_loader, train_loader, self.model_snn, self.args)
                state_a = self.infer(test_loader, train_loader, self.model_ann, self.args)
                acc_s = state_s['acc_test']
                acc_a = state_a['acc_test']
                acc_s_list.append(acc_s)
                acc_a_list.append(acc_a)
                F_s_list.append(state_s['F_test'])
                F_a_list.append(state_a['F_test'])
                R_s_list.append(state_s['R_test'])
                R_a_list.append(state_a['R_test'])
                Pre_s_list.append(state_s['Pre_test'])
                Pre_a_list.append(state_a['Pre_test'])
                fitness_s = state_s['acc_sum']
                fitness_a = state_a['acc_sum']
                fitness_s_list.append(fitness_s)
                fitness_a_list.append(fitness_a)
                topk_s_list.append(state_s['net'])
                topk_a_list.append(state_a['net'])
                CM_s.append(state_s['CM'])
                score_s.append(state_s['score'])
                label_s.append(state_s['label'])
                CM_a.append(state_a['CM'])
                score_a.append(state_a['score'])
                label_a.append(state_a['label'])
                print(f'first_acc_s: {acc_s:.4f}, first_acc_a: {acc_a:.4f}, first_fitness_s: {fitness_s:.4f}, first_fitness_a: {fitness_a:.4f}')
            else:
                # prepare next pool
                mut_pool_s, mut_pool_a, mut_acc_s, mut_acc_a, mut_F_s, mut_F_a, mut_R_s, mut_R_a, mut_Pre_s, mut_Pre_a, mut_fitness_s, mut_fitness_a, mut_CM_s, mut_CM_a, mut_score_s, mut_score_a, mut_label_s, mut_label_a = self._mutation(cand_s, cand_a, topk_pool_s, topk_pool_a, topk_fitness_s, topk_fitness_a, self.num_mutation, test_loader, train_loader, rate)
                acc_s_list += mut_acc_s
                F_s_list += mut_F_s
                R_s_list += mut_R_s
                Pre_s_list += mut_Pre_s
                fitness_s_list += mut_fitness_s
                topk_s_list += mut_pool_s
                acc_a_list += mut_acc_a
                F_a_list += mut_F_a
                R_a_list += mut_R_a
                Pre_a_list += mut_Pre_a
                fitness_a_list += mut_fitness_a
                topk_a_list += mut_pool_a
                CM_s += mut_CM_s
                CM_a += mut_CM_a
                score_s += mut_score_s
                score_a += mut_score_a
                label_s += mut_label_s
                label_a += mut_label_a

                cro_pool_s, cro_pool_a, cro_acc_s, cro_acc_a, cro_F_s, cro_F_a, cro_R_s, cro_R_a, cro_Pre_s, cro_Pre_a, cro_fitness_s, cro_fitness_a, cro_CM_s, cro_CM_a, cro_score_s, cro_score_a, cro_label_s, cro_label_a = self._crossover(cand_s, cand_a, topk_pool_s, topk_pool_a, topk_fitness_s, topk_fitness_a, self.num_crossover, test_loader, train_loader, rate)
                acc_s_list += cro_acc_s
                F_s_list += cro_F_s
                R_s_list += cro_R_s
                Pre_s_list += cro_Pre_s
                fitness_s_list += cro_fitness_s
                topk_s_list += cro_pool_s
                acc_a_list += cro_acc_a
                F_a_list += cro_F_a
                R_a_list += cro_R_a
                Pre_a_list += cro_Pre_a
                fitness_a_list += cro_fitness_a
                topk_a_list += cro_pool_a
                CM_s += cro_CM_s
                CM_a += cro_CM_a
                score_s += cro_score_s
                score_a += cro_score_a
                label_s += cro_label_s
                label_a += cro_label_a

            # get top-k candidates
            tmp_pool_s = topk_pool_s + topk_s_list
            tmp_pool_a = topk_pool_a + topk_a_list
            tmp_acc_s = topk_acc_s + acc_s_list
            tmp_acc_a = topk_acc_a + acc_a_list
            tmp_F_s = topk_F_s + F_s_list
            tmp_F_a = topk_F_a + F_a_list
            tmp_R_s = topk_R_s + R_s_list
            tmp_R_a = topk_R_a + R_a_list
            tmp_Pre_s = topk_Pre_s + Pre_s_list
            tmp_Pre_a = topk_Pre_a + Pre_a_list
            tmp_fitness_s = topk_fitness_s + fitness_s_list
            tmp_fitness_a = topk_fitness_a + fitness_a_list
            tmp_CM_s = topk_CM_s + CM_s
            tmp_CM_a = topk_CM_a + CM_a
            tmp_score_s = topk_score_s + score_s
            tmp_score_a = topk_score_a + score_a
            tmp_label_s = topk_label_s + label_s
            tmp_label_a = topk_label_a + label_a

            topk_idx_s = np.argsort(tmp_fitness_s)[::-1][:self.num_topk] ## decreasing order: the first is the highest
            topk_idx_a = np.argsort(tmp_fitness_a)[::-1][:self.num_topk]
            arch_acc_fitness_s = [[tmp_pool_s[idx], tmp_acc_s[idx], tmp_F_s[idx], tmp_R_s[idx], tmp_Pre_s[idx], tmp_fitness_s[idx], tmp_CM_s[idx], tmp_score_s[idx], tmp_label_s[idx]] for idx in topk_idx_s]
            arch_acc_fitness_a = [[tmp_pool_a[idx], tmp_acc_a[idx], tmp_F_a[idx], tmp_R_a[idx], tmp_Pre_a[idx], tmp_fitness_a[idx], tmp_CM_a[idx], tmp_score_a[idx], tmp_label_a[idx]] for idx in topk_idx_a]
            topk_pool_s = [tmp_pool_s[idx] for idx in topk_idx_s]
            topk_pool_a = [tmp_pool_a[idx] for idx in topk_idx_a]
            topk_acc_s = [tmp_acc_s[idx] for idx in topk_idx_s]
            topk_acc_a = [tmp_acc_a[idx] for idx in topk_idx_a]
            topk_F_s = [tmp_F_s[idx] for idx in topk_idx_s]
            topk_F_a = [tmp_F_a[idx] for idx in topk_idx_a]
            topk_R_s = [tmp_R_s[idx] for idx in topk_idx_s]
            topk_R_a = [tmp_R_a[idx] for idx in topk_idx_a]
            topk_Pre_s = [tmp_Pre_s[idx] for idx in topk_idx_s]
            topk_Pre_a = [tmp_Pre_a[idx] for idx in topk_idx_a]
            topk_fitness_s = [tmp_fitness_s[idx] for idx in topk_idx_s]
            topk_fitness_a = [tmp_fitness_a[idx] for idx in topk_idx_a]
            topk_CM_s = [tmp_CM_s[idx] for idx in topk_idx_s]
            topk_CM_a = [tmp_CM_a[idx] for idx in topk_idx_a]
            topk_score_s = [tmp_score_s[idx] for idx in topk_idx_s]
            topk_score_a = [tmp_score_a[idx] for idx in topk_idx_a]
            topk_label_s = [tmp_label_s[idx] for idx in topk_idx_s]
            topk_label_a = [tmp_label_a[idx] for idx in topk_idx_a]

            print(f'\t top-{self.num_topk} paths')
            for acc_s, acc_a, fitness_s, fitness_a in zip(topk_acc_s, topk_acc_a, topk_fitness_s, topk_fitness_a):
                avg_topk_s.update(acc_s)
                avg_topk_a.update(acc_a)
                avg_topk_fitness_s.update(fitness_s)
                avg_topk_fitness_a.update(fitness_a)
                print(f'\t top_acc_s: {acc_s:.4f}, top_acc_a: {acc_a:.4f}, '
                      f'\t top_fitness_s: {fitness_s:.4f}, top_fitness_a: {fitness_a:.4f}')
            print(f'\t avg_acc_s: {avg_topk_s.avg:.4f}, avg_acc_a: {avg_topk_a.avg:.4f}, '
                  f'\t avg_fitness_s: {avg_topk_fitness_s.avg:.4f}, avg_fitness_a: {avg_topk_fitness_a.avg:.4f}')

            # save history
            self.history_topk_s.append(arch_acc_fitness_s)
            self.history_topk_a.append(arch_acc_fitness_a)

        # return the history
        return self.history_topk_s, self.history_topk_a

    def infer(self, loader_test, loader_train, net, args):
        top1_test = AverageMeter()
        top1_train = AverageMeter()
        top1_sum = AverageMeter()
        F_test = AverageMeter()
        R_test = AverageMeter()
        Pre_test = AverageMeter()
        score_list = []
        label_list = []
        net.eval()
        confusion_test = ConfusionMatrix(num_classes=args.num_classes, args=args)
        confusion_train = ConfusionMatrix(num_classes=args.num_classes, args=args)

        with torch.no_grad():
            for step, (input_test, input_train) in enumerate(zip(loader_test, loader_train)):
                input_test, target_test = input_test
                input_train, target_train = input_train
                input_test, input_train = input_test.float().to(args.device), input_train.float().to(args.device)
                target_test, target_train = target_test.to(args.device), target_train.to(args.device)
                target_onehot = F.one_hot(target_test, args.num_classes).float()

                logits_test = net(input_test).to(args.device)
                logits_train = net(input_train).to(args.device)
                pred_test = logits_test.max(1, keepdim=True)[1]
                pred_train = logits_train.max(1, keepdim=True)[1]
                confusion_test.update(pred_test.cpu(), target_test.cpu())
                confusion_train.update(pred_train.cpu(), target_train.cpu())
                score_list.extend(logits_test.cpu().numpy())
                label_list.extend(target_onehot.cpu().numpy())

            acc_test, F1_test, Recall_test, Precision_test = confusion_test.summary()
            acc_train, F1_train, Recall_train, Precision_train = confusion_train.summary()
            top1_test.update(acc_test)
            top1_train.update(acc_train)
            F_test.update(F1_test)
            R_test.update(Recall_test)
            Pre_test.update(Precision_test)
            score_list = np.array(score_list)
            label_list = np.array(label_list)
            top1_sum.update((1 - args.fit) * acc_train + args.fit * acc_test)
            state = {
                'net': net.state_dict(),
                'acc_test': top1_test.avg,
                'acc_train': top1_train.avg,
                'acc_sum': top1_sum.avg,
                'F_test': F_test.avg,
                'R_test': R_test.avg,
                'Pre_test': Pre_test.avg,
                'CM': confusion_test,
                'score': score_list,
                'label': label_list
            }

        return state

