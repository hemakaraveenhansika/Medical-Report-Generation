import time
import pickle
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
from transformers import BertTokenizer

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger

import matplotlib.pyplot as plt


# feature_extractor_base
class DebuggerEncoderBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_train_loss = 10000000000

        self.temperature = 0.1
        self.use_cosine_similarity = True
        self.alpha_weight = 0.75

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.params = None

        self._init_encoder_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        self.bert_tokenizer = self._init_bert_tokenizer()
        self.encoder_state_dict = self.load_encoder_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform)

        self.extractor = self._init_visual_extractor()
        self.bert_encoder = self._init_bert_encoder()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        self.nt_xent_criterion = self._init_nt_xent()

        self.writer.write("{}\n".format(self.args))


    def encoder_train(self):
        print("train encoder start")
        results = {}
        for epoch_id in range(self.start_epoch, self.args.epochs):
            print('\n', f'Epoch {epoch_id}')

            train_contrastive_loss = self._epoch_encoder_train()
            val_contrastive_loss = self._epoch_encoder_validate()

            if self.args.mode == 'train':
                self.scheduler.step(train_contrastive_loss)
            else:
                self.scheduler.step(val_contrastive_loss)
            self.writer.write( "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(), epoch_id, train_contrastive_loss, val_contrastive_loss, self.optimizer.param_groups[0]['lr']))
            self.save_encoder(epoch_id, val_contrastive_loss, train_contrastive_loss)

            results[epoch_id] = {
                'epoch_id': epoch_id,
                'train_contrastive_loss': train_contrastive_loss,
                'val_contrastive_loss':val_contrastive_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }

        # print(results)
        self.__save_json(results)
        self.__plot_graph(results)
        print("train encoder done")

    def __save_json(self, result):
        result_path = self.args.result_path

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_encoder_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)

    def __plot_graph(self, result):
        keys = ['contrastive_loss']
        modes = ['train', 'val']
        for key in keys:
            for mode in modes:
                x = []
                y = []
                for i in range(self.start_epoch, self.args.epochs):
                    print(str(i), mode+'_'+key)
                    x.append(i)
                    y.append(result[i][mode+'_'+key])
                plt.plot(x, y, label=mode)
                plt.xlabel('epoch')
                plt.ylabel(key)
                plt.title(key)
            plt.legend()
            plt.show()

    def _epoch_encoder_train(self):
        raise NotImplementedError

    def _epoch_encoder_validate(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.encoder_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # model_dir = os.path.join(model_dir, self._get_now())

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(model_dir)
        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab

    def _init_bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        return tokenizer

    def load_encoder_state_dict(self):
        self.start_epoch = 0
        try:
            encoder_state = torch.load(self.args.load_encoder_path)
            self.start_epoch = encoder_state['epoch'] + 1
            self.writer.write("[Load Encoder-{} Succeed!]\n".format(self.args.load_encoder_path))
            self.writer.write("Load From Epoch {}\n".format(encoder_state['epoch']))
            print("Load From Epoch {}".format(encoder_state['epoch']))
            return encoder_state
        except Exception as err:
            self.writer.write("[Load Encoder Failed] {}\n".format(err))
            print("[Load Encoder Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)
        try:
            model.load_state_dict(self.encoder_state_dict['extractor'])
            print("Visual Extractor Loaded!")
            self.writer.write("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()

        return model

    def _init_bert_encoder(self):
        model = BertClassfier(bert_base_model='bert-base-uncased', out_dim=1920, freeze_layers=[0,1,2,3,4,5])

        try:
            model.load_state_dict(self.encoder_state_dict['bert'])
            print("Bert_encoder Loaded!")
            self.writer.write("[Load BERT model Succeed!]\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load BERT model Failed {}!]\n".format(err))

        if not self.args.bert_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model


    def _init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader


    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train loss': train_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            print(tag, value, epoch + 1)
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.args.result_path, 'logs'))
        return logger

    def _init_nt_xent(self):
        nt_xent = NTXentLoss(self.device, self.args.batch_size, self.temperature, self.use_cosine_similarity, self.alpha_weight)
        if self.args.cuda:
            nt_xent = nt_xent.cuda()
        return nt_xent

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_encoder_path(self):
        if not os.path.exists(self.args.encoder_path):
            os.makedirs(self.args.encoder_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def save_encoder(self, epoch_id, val_loss, train_loss):
        def save_whole_encoder(_filename):
            self.writer.write("Saved Encoder Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'bert': self.bert_encoder.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))
            print("save whole model in", os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_encoder(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value}, os.path.join(self.model_dir, "{}".format(_filename)))

        if val_loss < self.min_val_loss:
            file_name = "val_encoder_best_loss.pth.tar"
            save_whole_encoder(file_name)
            self.min_val_loss = val_loss

        if train_loss < self.min_train_loss:
            file_name = "train_encoder_best_loss.pth.tar"
            save_whole_encoder(file_name)
            self.min_train_loss = train_loss


# model_base
class DebuggerModelBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_val_tag_loss = 1000000
        self.min_val_stop_loss = 1000000
        self.min_val_word_loss = 10000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000
        self.min_train_stop_loss = 1000000
        self.min_train_word_loss = 10000000

        # self.temperature = 0.1
        # self.use_cosine_similarity = True
        # self.alpha_weight = 0.75

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        # self.bert_tokenizer = self._init_bert_tokenizer()
        self.model_state_dict = self.load_model_state_dict()
        self.encoder_state_dict = self.load_encoder_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform)

        self.extractor = self._init_visual_extractor()
        # self.bert_encoder = self._init_bert_encoder()
        self.mlc = self._init_mlc()
        self.co_attention = self._init_co_attention()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        # self.nt_xent_criterion = self._init_nt_xent()

        self.writer.write("{}\n".format(self.args))


    def model_train(self):
        print("train model start")
        results = {}
        for epoch_id in range(self.start_epoch, self.args.epochs):
            print('\n', f'Epoch {epoch_id}')

            train_tag_loss, train_stop_loss, train_word_loss, train_loss = self._epoch_model_train()
            val_tag_loss, val_stop_loss, val_word_loss, val_loss = self._epoch_model_validate()

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step(val_loss)
            self.writer.write( "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(), epoch_id, train_loss, val_loss, self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id, val_loss, val_tag_loss, val_stop_loss, val_word_loss, train_loss)

            results[epoch_id] = {
                'epoch_id': epoch_id,
                'train_tags_loss': train_tag_loss,
                'train_stop_loss': train_stop_loss,
                'train_word_loss': train_word_loss,
                'train_loss': train_loss,
                'val_tags_loss': val_tag_loss,
                'val_stop_loss': val_stop_loss,
                'val_word_loss': val_word_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }

            # self._log(train_tags_loss=train_tag_loss,
            #           train_stop_loss=train_stop_loss,
            #           train_word_loss=train_word_loss,
            #           train_loss=train_loss,
            #           val_tags_loss=val_tag_loss,
            #           val_stop_loss=val_stop_loss,
            #           val_word_loss=val_word_loss,
            #           val_loss=val_loss,
            #           lr=self.optimizer.param_groups[0]['lr'],
            #           epoch=epoch_id)

            # print("train_loss, val_loss ", epoch_id, train_loss, val_loss)
        print(results)
        self.__save_json(results)
        self.__plot_graph(results)
        print("train done")

    def __save_json(self, result):
        # result_path = os.path.join(self.args.model_dir, self.args.result_path)
        # result_path = "/kaggle/working/Medical-Report-Generation/results"
        result_path = self.args.result_path

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_model_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)

    def __plot_graph(self, result):
        keys = ['loss']
        modes = ['train', 'val']
        for key in keys:
            for mode in modes:
                x = []
                y = []
                for i in range(self.start_epoch, self.args.epochs):
                    print(str(i), mode+'_'+key)
                    x.append(i)
                    y.append(result[i][mode+'_'+key])
                plt.plot(x, y, label=mode)
                plt.xlabel('epoch')
                plt.ylabel(key)
                plt.title(key)
            plt.legend()
            plt.show()

    def _epoch_model_train(self):
        raise NotImplementedError

    def _epoch_model_validate(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # model_dir = os.path.join(model_dir, self._get_now())

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(model_dir)
        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab

    # def _init_bert_tokenizer(self):
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     return tokenizer

    def load_encoder_state_dict(self):
        try:
            encoder_state = torch.load(self.args.load_encoder_path)
            self.writer.write("[Load Encoder-{} Succeed!]\n".format(self.args.load_encoder_path))
            print("[Load Encoder-{} Succeed!]".format(self.args.load_encoder_path))
            return encoder_state
        except Exception as err:
            self.writer.write("[Load Encoder Failed] {}\n".format(err))
            print("[Load Encoder Failed] {}\n".format(err))
            return None

    def load_model_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch'] + 1
            self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            print("Load From Epoch {}".format(model_state['epoch']))
            return model_state
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))
            print("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)
        try:
            # model_state = torch.load(self.args.load_visual_model_path)
            # model.load_state_dict(model_state['model'])
            model.load_state_dict(self.encoder_state_dict['extractor'])
            print("Visual Extractor Loaded!")
            self.writer.write("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()

        return model

    # def _init_bert_encoder(self):
    #     model = BertClassfier(bert_base_model='bert-base-uncased', out_dim=1920, freeze_layers=[0,1,2,3,4,5])
    # 
    #     try:
    #         # model_state = torch.load(self.args.load_mlc_model_path)
    #         # model.load_state_dict(model_state['model'])
    #         model.load_state_dict(self.encoder_state_dict['bert'])
    #         print("Bert_encoder Loaded!")
    #         self.writer.write("[Load BERT model Succeed!]\n")
    #     except Exception as err:
    #         print(err)
    #         self.writer.write("[Load BERT model Failed {}!]\n".format(err))
    # 
    #     if not self.args.bert_trained:
    #         for i, param in enumerate(model.parameters()):
    #             param.requires_grad = False
    #     else:
    #         if self.params:
    #             self.params += list(model.parameters())
    #         else:
    #             self.params = list(model.parameters())
    # 
    #     if self.args.cuda:
    #         model = model.cuda()
    #     return model

    def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        try:
            # model_state = torch.load(self.args.load_mlc_model_path)
            # model.load_state_dict(model_state['model'])
            model.load_state_dict(self.model_state_dict['mlc'])
            print("Mlc Loaded!")
            self.writer.write("[Load MLC Succeed!]\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load MLC Failed {}!]\n".format(err))

        if not self.args.mlc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        try:
            # model_state = torch.load(self.args.load_co_model_path)
            # model.load_state_dict(model_state['model'])
            model.load_state_dict(self.model_state_dict['co_attention'])
            print("co_attention Loaded!")
            self.writer.write("[Load Co-attention Succeed!]\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load Co-attention Failed {}!]\n".format(err))

        if not self.args.co_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_sentence_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    def _init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             val_tags_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'val tags loss': val_tags_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            print(tag, value, epoch + 1)
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.args.result_path, 'logs'))
        return logger

    # def _init_nt_xent(self):
    #     nt_xent = NTXentLoss(self.device, self.args.batch_size, self.temperature, self.use_cosine_similarity, self.alpha_weight)
    #     if self.args.cuda:
    #         nt_xent = nt_xent.cuda()
    #     return nt_xent

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    val_loss,
                    val_tag_loss,
                    val_stop_loss,
                    val_word_loss,
                    train_loss):
        def save_whole_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            print("save whole model in", os.path.join(self.model_dir, "{}".format(_filename)))
            torch.save({
                        # 'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        # 'bert': self.bert_encoder.state_dict(),
                        'co_attention': self.co_attention.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if val_loss < self.min_val_loss:
            file_name = "val_model_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_val_loss = val_loss

        if train_loss < self.min_train_loss:
            file_name = "train_model_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss

        # if val_tag_loss < self.min_val_tag_loss:
        #     save_part_model("extractor.pth.tar", self.extractor.state_dict())
        #     save_part_model("mlc.pth.tar", self.mlc.state_dict())
        #     self.min_val_tag_loss = val_tag_loss
        #
        # if val_stop_loss < self.min_val_stop_loss:
        #     save_part_model("sentence.pth.tar", self.sentence_model.state_dict())
        #     self.min_val_stop_loss = val_stop_loss
        #
        # if val_word_loss < self.min_val_word_loss:
        #     save_part_model("word.pth.tar", self.word_model.state_dict())
        #     self.min_val_word_loss = val_word_loss


# encoder train-validation
class ContrastiveModel(DebuggerEncoderBase):
    def _init_(self, args):
        DebuggerEncoderBase.__init__(self, args)
        self.args = args

    def _epoch_encoder_train(self):
        contrastive_loss = 0
        self.extractor.train()
        self.bert_encoder.train()

        for images, image_id, label, captions, prob, text in tqdm(self.train_data_loader):

            batch_contrastive_loss = 0
            images = self._to_var(images)
            bert_tokens = self.bert_tokenizer(list(text), return_tensors="pt", padding=True, truncation=True)
            bert_tokens = bert_tokens.to('cuda' if torch.cuda.is_available() else 'cpu')


            visual_features, avg_features = self.extractor.forward(images)
            text_features = self.bert_encoder.forward(bert_tokens)

            batch_contrastive_loss = self.nt_xent_criterion(avg_features, text_features)

            print("\navg_features.shape", avg_features.shape)
            print("text_features.shape", text_features.shape)
            print("\nbatch contrastive loss :", batch_contrastive_loss.item())

            self.optimizer.zero_grad()
            batch_contrastive_loss.backward()
            self.optimizer.step()
            contrastive_loss += batch_contrastive_loss.item()

        return contrastive_loss

    def _epoch_encoder_validate(self):
        contrastive_loss = 0
        self.extractor.eval()
        self.bert_encoder.eval()

        for images, image_id, label, captions, prob, text in tqdm(self.val_data_loader):

            batch_contrastive_losss = 0
            images = self._to_var(images, requires_grad=False)
            bert_tokens = self.bert_tokenizer(list(text), return_tensors="pt", padding=True, truncation=True)
            bert_tokens = bert_tokens.to('cuda' if torch.cuda.is_available() else 'cpu')

            visual_features, avg_features = self.extractor.forward(images)
            text_features = self.bert_encoder.forward(bert_tokens)
            
            batch_contrastive_loss = self.nt_xent_criterion(avg_features, text_features)
            contrastive_loss += batch_contrastive_loss.item()

        return contrastive_loss


# model train-validation

class LSTMDebugger(DebuggerModelBase):
    def _init_(self, args):
        DebuggerModelBase.__init__(self, args)
        self.args = args

    def _epoch_model_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        # self.extractor.train()
        self.extractor.eval()
        # self.bert_encoder.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()

        # for i, (images, _, label, captions, prob) in enumerate(self.train_data_loader):
        for images, image_id, label, captions, prob, text in tqdm(self.train_data_loader):

            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images)

            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            visual_features, avg_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)

            print("\nvisual_features.shape", visual_features.shape)
            print("avg_features.shape", avg_features.shape)
            print("semantic_features.shape", semantic_features.shape)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            for sentence_index in range(captions.shape[1]):
                ctx, _, _ = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)
                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx, prev_hidden_states, sentence_states)
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                # print("p_stop:{}".format(p_stop.squeeze()))
                # print("prob_real:{}".format(prob_real[:, sentence_index]))

                for word_index in range(1, captions.shape[2]):
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index]) * word_mask).sum() * (0.9 ** word_index)

                    # batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])).sum()
                    # print("words:{}".format(torch.max(words, 1)[1]))
                    # print("real:{}".format(context[:, sentence_index, word_index]))

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            print("batch_tag loss :", self.args.lambda_tag * batch_tag_loss.item())
            print("batch_stop loss :", self.args.lambda_stop * batch_stop_loss.item())
            print("batch_word loss :", self.args.lambda_word * batch_word_loss.item())
            print("\nbatch loss :", batch_loss.item())

            self.optimizer.zero_grad()
            batch_loss.backward()

            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)

            self.optimizer.step()

            tag_loss += self.args.lambda_tag * batch_tag_loss.item()
            stop_loss += self.args.lambda_stop * batch_stop_loss.item()
            word_loss += self.args.lambda_word * batch_word_loss.item()
            loss += batch_loss.item()

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_model_validate(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        # for i, (images, _, label, captions, prob) in enumerate(self.val_data_loader):
        for images, image_id, label, captions, prob, text in tqdm(self.val_data_loader):
            
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images, requires_grad=False)
            
            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
            
            visual_features, avg_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)
            
            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            for sentence_index in range(captions.shape[1]):
                ctx, v_att, a_att = self.co_attention.forward(avg_features,
                                                              semantic_features,
                                                              prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)
                # print("p_stop:{}".format(p_stop.squeeze()))
                # print("prob_real:{}".format(prob_real[:, sentence_index]))

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                for word_index in range(1, captions.shape[2]):
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
                                        * word_mask).sum()
                    # print("words:{}".format(torch.max(words, 1)[1]))
                    # print("real:{}".format(context[:, sentence_index, word_index]))

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            tag_loss += self.args.lambda_tag * batch_tag_loss.item()
            stop_loss += self.args.lambda_stop * batch_stop_loss.item()
            word_loss += self.args.lambda_word * batch_word_loss.item()
            loss += batch_loss.item()

        return tag_loss, stop_loss, word_loss, loss

    def _init_sentence_model(self):
        model = SentenceLSTM(version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        try:
            # model_state = torch.load(self.args.load_sentence_model_path)
            # model.load_state_dict(model_state['model'])
            model.load_state_dict(self.model_state_dict['sentence_model'])
            print("Sentence Model Loaded!")
            self.writer.write("[Load Sentence Model Succeed!\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load Sentence model Failed {}!]\n".format(err))

        if not self.args.sentence_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        try:
            # model_state = torch.load(self.args.load_word_model_path)
            # model.load_state_dict(model_state['model'])
            model.load_state_dict(self.model_state_dict['word_model'])
            print("Word Model Loaded!")
            self.writer.write("[Load Word Model Succeed!\n")
        except Exception as err:
            print(err)
            self.writer.write("[Load Word model Failed {}!]\n".format(err))

        if not self.args.word_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.cuda()
        return model


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab.pkl', help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/chest-xrays-indiana-university/images/images_normalized', help='the path for images')
    # parser.add_argument('--image_dir', type=str, default='/content/drive/MyDrive/FYP17-captioning/Datasets/iu/images/images_normalized', help='the path for images')

    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/new_data/train_data.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/new_data/val_data.txt',
                        help='the val array')
    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--encoder_path', type=str, default='./models/', help='path for saving trained encoder')
    parser.add_argument('--load_encoder_path', type=str, default='/kaggle/input/mrg-contrastive-model/Medical-Report-Generation/models/v4/train_encoder_best_loss.pth.tar', help='The path of loaded encoder')
    parser.add_argument('--load_model_path', type=str, default='', help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='v4', help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='densenet201', help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False, help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str, default='.')
    parser.add_argument('--visual_trained', action='store_true', default=True, help='Whether train visual extractor or not')

    #BERT
    parser.add_argument('--bert_trained', action='store_true', default=True)

    # MLC
    parser.add_argument('--classes', type=int, default=210)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--load_mlc_model_path', type=str, default='.')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_co_model_path', type=str, default='.')
    parser.add_argument('--co_trained', action='store_true', default=True)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--load_sentence_model_path', type=str, default='.')
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)
    parser.add_argument('--load_word_model_path', type=str,
                        default='.')
    parser.add_argument('--word_trained', action='store_true', default=True)

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=8)

    parser.add_argument('--clip', type=float, default=-1, help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_contrast', type=float, default=100)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    # Saved result
    parser.add_argument('--result_path', type=str, default='results',
                        help='the path for storing results')
    parser.add_argument('--result_model_name', type=str, default='logs_model',
                        help='the name of results')
    parser.add_argument('--result_encoder_name', type=str, default='logs_encoder',
                        help='the name of results')

    parser.add_argument('--phase', type=str, default='encoder', help='the name of base')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    if(args.phase == 'decoder'):
        print("run decoder")
        lstm_debugger = LSTMDebugger(args)
        lstm_debugger.model_train()
    else:
        print("run encoder")
        contrastive_model = ContrastiveModel(args)
        contrastive_model.encoder_train()
    
    
