# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import logging
from scipy.io import loadmat
import csv
# Our libs


from . dataset import TestDataset
from . models import ModelBuilder, SegmentationModule
from . utils import colorEncode, find_recursive, setup_logger
from . lib.nn import user_scattered_collate, async_copy_to
from . lib.utils import as_numpy

from PIL import Image
from tqdm import tqdm
from . config import cfg
import sys

class SegmentationProxy():
    def __init__(self, argsList):
        # Setup the parameters
        #-------------------------------------
        args = self.parseArguments(argsList)

        # Setup some visualization local tables
        dataPath = os.path.join(args.resourcesFolderPath, "data")
        self.colors = loadmat(os.path.join(dataPath, "color150.mat"))['colors']
        self.names = {}
        with open(os.path.join(dataPath, 'object150_info.csv')) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

        self.resourcesFolderPath = args.resourcesFolderPath

        assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
            'PyTorch>=0.4.0 is required'

        cfg.merge_from_file(args.cfg)
        cfg.merge_from_list(args.opts)
        # cfg.freeze()

        self.logger = setup_logger(distributed_rank=0)  # TODO
        self.logger.info("Loaded configuration file {}".format(args.cfg))
        self.logger.info("Running with config:\n{}".format(cfg))

        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        # absolute paths of model weights
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
               os.path.exists(cfg.MODEL.weights_decoder), f"checkpoint {cfg.MODEL.weights_encoder} does not exitst!"

        # generate testing image list
        if os.path.isdir(args.imgs):
            imgs = find_recursive(args.imgs, ext=args.extension)
        else:
            imgs = [args.imgs]
        assert len(imgs), "imgs should be a path to image (.jpg) or directory."
        cfg.list_test = sorted([{'fpath_img': x} for x in imgs], key = lambda record : record['fpath_img'])

        if not os.path.isdir(cfg.TEST.result):
            os.makedirs(cfg.TEST.result)

        if not os.path.isdir(cfg.TEST.resultComp):
            os.makedirs(cfg.TEST.resultComp)
        #-------------------------------------

        # Init the internal segmentation module
        self.segmentation_module = None
        self.initModel(cfg, args.gpu)

    def parseArguments(self, argsList):
        parser = argparse.ArgumentParser(
            description="PyTorch Semantic Segmentation Testing"
        )
        parser.add_argument(
            "--imgs",
            required=True,
            type=str,
            help="an image paths, or a directory name"
        )
        '''
        parser.add_argument(
            "--scaleFactor",
            default=1.0,
            type=float,
            help="gpu id for evaluation"
        )
        '''
        parser.add_argument(
            "--cfg",
            default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
        parser.add_argument(
            "--gpu",
            default=0,
            type=int,
            help="gpu id for evaluation"
        )

        parser.add_argument(
            "--resourcesFolderPath",
            default=None,
            type=str,
            help="path to where metadata resources needed are"
        )

        parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )

        parser.add_argument(
            "--extension",
            help="Extensions",
            default='.png',
            type=str,
        )

        segArgs = argsList if isinstance(argsList, list) else argsList.split()
        #seqArgs = ["--imgs", 'semanticSegmentation/TEST_INPUT/10023947602400723454_1120_000_1140_000'].extend(segArgs)
        aargs = parser.parse_args(segArgs)

        #print(aargs.imgs)
        return aargs

    def initModel(self, cfg, gpu):
        if gpu is not None and gpu >= 0 and gpu != "-1":
            print("Setting gpu device to ", gpu, type(gpu))
            torch.cuda.set_device(gpu)

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        if gpu is not None and gpu > -1:
            self.segmentation_module.cuda()

        self.cfg = cfg
        self.gpu = gpu
        if self.gpu == "-1" or self.gpu == None:
            self.gpu = -1

        # Dataset and Loader
        dataset_test = TestDataset(
            cfg.list_test,
            cfg.DATASET)
        self.loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=0,
            drop_last=True)

    # This function returns a dictiory from outputDictExtractor(fileName)->pred array - for each pixel and every file in the dictionary
    # outputDictExtractor is used to extract from file input name to the prediction
    def doInference(self, outputDictExtractor = None, filterFunctor = None):
        self.segmentation_module.eval()
        outputFolder = cfg.TEST.result

        NumItemsInData = len(self.loader)
        pbar = tqdm(total=NumItemsInData)
        for batch_data in self.loader:
            # process data
            batch_data = batch_data[0]
            imgPath = batch_data["info"]
            if filterFunctor is not None:
                if not filterFunctor(imgPath, outputFolder):
                    self.logger.log(logging.DEBUG, (f"Skipping {imgPath[imgPath.rfind('/'):]} already done or not needed"))
                    continue

            segSize = (batch_data['img_ori'].shape[0],
                       batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']

            with torch.no_grad():
                scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])

                if self.gpu != -1:
                    scores = async_copy_to(scores, self.gpu)

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    if self.gpu != -1:
                        feed_dict = async_copy_to(feed_dict, self.gpu)

                    # forward pass
                    pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())

                # Scale back predictions to the original image format
                if cfg.DATASET.scaleFactor > 1:
                    SF = int(cfg.DATASET.scaleFactor)
                    pred = np.kron(pred, np.ones((SF, SF)))

                #if outputDictExtractor is not None:
                outputKey = outputDictExtractor(batch_data["info"], pred, cfg.TEST.result)

            if cfg.TEST.saveOnlyLabels == 0:
                # visualization
                self.visualize_result(
                    (batch_data['img_ori'], batch_data['info']),
                    pred,
                    self.cfg
                )

            pbar.update(1)

    def visualize_result(self, data, pred, cfg):
        (img, info) = data
        pred = np.int32(pred)

        if cfg.DATASET.scaleFactor > 1:
            SF = int(cfg.DATASET.scaleFactor)
            img = img.transpose((2, 0 ,1))
            img = np.kron(img, np.ones((SF, SF)))
            img = img.transpose((1, 2, 0)).astype(dtype=np.uint8)
        """
        # print predictions in descending order
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))
        """
        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        # aggregate images and save
        im_vis = np.concatenate((img, pred_color), axis=1)

        img_name = info.split('/')[-1]
        Image.fromarray(im_vis).save(
            os.path.join(cfg.TEST.resultComp, img_name))

        Image.fromarray(pred_color).save(
            os.path.join(cfg.TEST.result, img_name))



def runTestSample(args, extractOutputFunctor, filterFunctor, forceRecompute = False):
    segmentationProxy = SegmentationProxy(args)
    segmentationProxy.doInference(extractOutputFunctor, filterFunctor)

if __name__ == '__main__':
    segmentationProxy = SegmentationProxy(sys.argv[1:])


