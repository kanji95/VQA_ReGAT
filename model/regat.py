"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from model.fusion import BAN, BUTD, MuTAN
from model.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from model.classifier import SimpleClassifier


class ReGAT(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_embedding, count_embedding, classifier, q_classifier, glimpse, fusion, relation_type):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.count_embedding = count_embedding
        self.classifier = classifier
        self.q_classifier = q_classifier

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        batch_size = v.shape[0]
        
        w_emb = self.w_emb(q)
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)
        
        q_type = self.q_classifier(q_emb_self_att)
        
        # print(f'w_emb: {w_emb.shape}, q_emb_seq: {q_emb_seq.shape}, q_emb_self_att: {q_emb_self_att.shape}', flush=True)

        # [batch_size, num_rois, out_dim]
        if self.relation_type == "semantic":
            v_emb = self.v_relation.forward(v, sem_adj_matrix, q_emb_self_att)
        elif self.relation_type == "spatial":
            v_emb = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # implicit
            v_emb = self.v_relation.forward(v, implicit_pos_emb,
                                            q_emb_self_att)

        if self.fusion == "ban":
            joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
            count_emb, count_att = self.count_embedding(v_emb, q_emb_seq, b)
            
            _, indices = torch.max(q_type, dim=1)
            embeddings = torch.stack([joint_emb, count_emb, joint_emb], dim=1)
            joint_emb = embeddings[torch.arange(batch_size), indices]
            
        elif self.fusion == "butd":
            q_emb = self.q_emb(w_emb)  # [batch, q_dim]
            joint_emb, att = self.joint_embedding(v_emb, q_emb)
        else:  # mutan
            joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att)
        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb
        # print(f'logits: {logits.shape}, att: {att.shape}', flush=True)
        return q_type, logits, att


def build_regat(dataset, args):
    print("Building ReGAT model with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)

    if args.relation_type == "semantic":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "spatial":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    else:
        # dataset.v_dim = 2048, args.num_hid = 1024, args.relation_dim = 1024, args.dir_num = 2, args.imp_pos_emb_dim = 64, args.nongt_dim = 20,
        # args.num_heads = 16, args.num_steps = 1, args.residual_connection = true, args.label_bias = false
        
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads=args.num_heads, num_steps=args.num_steps,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    q_classifier = SimpleClassifier(args.num_hid, args.num_hid // 2, 3, 0.5)
    
    gamma = 0
    count_embedding = None
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma, use_counter=False)
        count_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma, use_counter=True)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    return ReGAT(dataset, w_emb, q_emb, q_att, v_relation, joint_embedding, count_embedding,
                 classifier, q_classifier, gamma, args.fusion, args.relation_type)
