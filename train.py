import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd as autograd
import numpy as np
import copy
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
from lib.utils.confuse_matrix import *
from lib.utils.Fourier_Aug import *
from train.utils import *
from train.loss import *
from train.context import disable_tracking_bn_stats
from train.ramps import exp_rampup

def train_MSCAD(train_dloader_list, test_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, 
            local_models, local_classifiers, local_optimizers, local_classifier_optimizers, epoch, writer,
            num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
            confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, args, pre_models=None, pre_classifiers=None):
    task_criterion = nn.CrossEntropyLoss().cuda()
    criterion_im = Lim(1e-5)
    source_domain_num = len(train_dloader_list[1:])
    # global model
    global_models = []
    global_classifiers = []
    for i in range(0, len(model_list)):
        # global_models.append(copy.deepcopy(model_list[i]))
        global_classifiers.append(copy.deepcopy(classifier_list[i]))
    # train mode
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    # train source domain
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                    model_list[1:],
                                                                                    classifier_list[1:],
                                                                                    optimizer_list[1:],
                                                                                    classifier_optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for idx_batch, (image_ws, label_s) in enumerate(train_dloader):
                if idx_batch >= batch_per_epoch:
                    break
                image_s = image_ws[0].cuda()
                image_s_s = image_ws[1].cuda()
                label_s = label_s.long().cuda()
                label_onehot = create_onehot(label_s, num_classes)
                label_onehot = label_onehot.cuda()
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                
                # Adversarial Style Augmentation
                # Block1
                image_s = model.forward_block1(image_s)

                # Get style feature and normalized image
                # print('image_s:{}'.format(image_s.size()))
                B = image_s.size(0)
                mu = image_s.mean(dim=[2, 3], keepdim=True)
                var = image_s.var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-5).sqrt()
                mu, sig = mu.detach(), sig.detach()
                input_normed = (image_s - mu) / sig
                input_normed = input_normed.detach().clone()

                # Set learnable style feature and adv optimizer
                adv_mu, adv_sig = mu, sig
                adv_mu.requires_grad_(True)
                adv_sig.requires_grad_(True)
                adv_optim = torch.optim.SGD(params=[adv_mu, adv_sig], lr=args.lr_adv, momentum=0, weight_decay=0)

                # Optimize adversarial style feature
                adv_optim.zero_grad()
                adv_input = input_normed * adv_sig + adv_mu

                # domain-specific adversarial loss
                for i in range(1, len(pre_classifiers)):
                    adv_output = pre_classifiers[i](model.forward_rest(model.forward_block2(adv_input)))
                    if i == 1:
                        adv_loss = torch.nn.functional.cross_entropy(adv_output, label_s)
                    else:
                        adv_loss += torch.nn.functional.cross_entropy(adv_output, label_s)
                
                all_loss = - adv_loss/(len(pre_classifiers)-1)
                all_loss.backward()
                adv_optim.step()

                ### Robust Model Training
                model.train()
                classifier.train()
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                adv_input = input_normed * adv_sig + adv_mu

                # Block2
                image_s = model.forward_block2(adv_input)

                # Get style feature and normalized image
                # print('image_s:{}'.format(image_s.size()))
                B = image_s.size(0)
                mu = image_s.mean(dim=[2, 3], keepdim=True)
                var = image_s.var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-5).sqrt()
                mu, sig = mu.detach(), sig.detach()
                input_normed = (image_s - mu) / sig
                input_normed = input_normed.detach().clone()

                # Set learnable style feature and adv optimizer
                adv_mu, adv_sig = mu, sig
                adv_mu.requires_grad_(True)
                adv_sig.requires_grad_(True)
                adv_optim = torch.optim.SGD(params=[adv_mu, adv_sig], lr=args.lr_adv, momentum=0, weight_decay=0)

                # Optimize adversarial style feature
                adv_optim.zero_grad()
                adv_input = input_normed * adv_sig + adv_mu

                # domain-specific adversarial loss
                for i in range(1, len(pre_classifiers)):
                    adv_output = pre_classifiers[i](model.forward_rest(adv_input))
                    if i == 1:
                        adv_loss = torch.nn.functional.cross_entropy(adv_output, label_s)
                    else:
                        adv_loss += torch.nn.functional.cross_entropy(adv_output, label_s)
                
                all_loss = - adv_loss/(len(pre_classifiers)-1)
                all_loss.backward()
                adv_optim.step()

                ### Robust Model Training
                model.train()
                classifier.train()
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                adv_input = input_normed * adv_sig + adv_mu

                # # Block3
                # image_s = model.forward_block3(adv_input)

                # # Get style feature and normalized image
                # # print('image_s:{}'.format(image_s.size()))
                # B = image_s.size(0)
                # mu = image_s.mean(dim=[2, 3], keepdim=True)
                # var = image_s.var(dim=[2, 3], keepdim=True)
                # sig = (var + 1e-5).sqrt()
                # mu, sig = mu.detach(), sig.detach()
                # input_normed = (image_s - mu) / sig
                # input_normed = input_normed.detach().clone()

                # # Set learnable style feature and adv optimizer
                # adv_mu, adv_sig = mu, sig
                # adv_mu.requires_grad_(True)
                # adv_sig.requires_grad_(True)
                # adv_optim = torch.optim.SGD(params=[adv_mu, adv_sig], lr=args.lr_adv, momentum=0, weight_decay=0)

                # # Optimize adversarial style feature
                # adv_optim.zero_grad()
                # adv_input = input_normed * adv_sig + adv_mu

                # # domain-specific adversarial loss
                # for i in range(1, len(pre_classifiers)):
                #     adv_output = pre_classifiers[i](model.forward_rest(adv_input))
                #     if i == 1:
                #         adv_loss = torch.nn.functional.cross_entropy(adv_output, label_s)
                #     else:
                #         adv_loss += torch.nn.functional.cross_entropy(adv_output, label_s)
                
                # all_loss = - adv_loss/(len(pre_classifiers)-1)
                # all_loss.backward()
                # adv_optim.step()

                # ### Robust Model Training
                # model.train()
                # classifier.train()
                # # reset grad
                # optimizer.zero_grad()
                # classifier_optimizer.zero_grad()

                # adv_input = input_normed * adv_sig + adv_mu
                
                # Domain-Invariant Learning
                output_ori = classifier(model.forward_rest(image_s))
                output_adv = classifier(model.forward_rest(adv_input))
                src_loss1 = F.cross_entropy(output_adv, label_s)
                src_loss2 = F.cross_entropy(output_ori, label_s)
                task_loss = src_loss1 + src_loss2

                # cross-domain relation matching
                
                p_ori, p_aug = F.softmax(output_ori / args.tau, dim=1), F.softmax(output_adv / args.tau, dim=1)
                with torch.no_grad():
                    for i in range(1, len(pre_classifiers)):
                        ori_temp = pre_classifiers[i](model.forward_rest(image_s))
                        if i == 1:
                            ensemble_logit = ori_temp
                        else:
                            ensemble_logit += ori_temp
                    ensemble_logit /= (len(pre_classifiers)-1)
                    ensemble_p = F.softmax(ensemble_logit / args.tau, dim=1)

                loss_inter_kd1 = F.kl_div(p_ori.log(), ensemble_p, reduction='batchmean')
                loss_inter_kd2 = F.kl_div(p_aug.log(), ensemble_p, reduction='batchmean')
                loss_cdrm = loss_inter_kd1 + loss_inter_kd2

                # supervised contrastive loss
                con_fn = SupConLoss()
                emb_src = F.normalize(model.forward_rest(image_s)).unsqueeze(1)
                emb_aug = F.normalize(model.forward_rest(adv_input)).unsqueeze(1)
                con_loss = con_fn(torch.cat([emb_src, emb_aug], dim=1), label_s)

                loss = task_loss + args.con * con_loss + args.cdrm * loss_cdrm
                
                loss.backward()
                optimizer.step()
                classifier_optimizer.step()
    
    # save local models
    pre_models = []
    pre_classifiers = []
    for i in range(0, len(model_list)):
        pre_models.append(copy.deepcopy(model_list[i]))
        pre_classifiers.append(copy.deepcopy(classifier_list[i]))

    ## model aggregate
    # domain weights of aggregation
    domain_weight = []
    num_domains = len(model_list[1:])
    for i in range(num_domains):
        domain_weight.append(1.0/num_domains)
    
    # fedavg
    federated_avg(model_list[1:], domain_weight, mode='fedavg')
    federated_avg(classifier_list[1:], domain_weight, mode='fedavg')

    return pre_models, pre_classifiers, domain_weight
