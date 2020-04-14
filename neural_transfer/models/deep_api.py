# -*- coding: utf-8 -*-
"""
Integrate a model with the DEEP API
"""

import json
import argparse
import pkg_resources
import os
import pickle

import neural_transfer.config as cfg
import neural_transfer.models.image_utils as iutils
import neural_transfer.models.file_utils as futils
import neural_transfer.models.style_transfer as transfer_style

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from aiohttp.web import HTTPBadRequest

from flaat import Flaat
#from __future__ import print_function
flaat = Flaat()

#def _catch_error(f):
#    def wrap(*args, **kwargs):
#        try:
#            return f(*args, **kwargs)
#        except Exception as e:
#            raise HTTPBadRequest(reason=e)


def _fields_to_dict(fields_in):
    """
    Example function to convert mashmallow fields to dict()
    """
    dict_out = {}
    
    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, 
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split('.', 1)

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, 
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    ### One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    ### One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name': None,
        'version': None,
        'summary': None,
        'home-page': None,
        'author': None,
        'author-email': None,
        'license': None,
        'help-train' : train_args,
        'help-predict' : predict_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # e.g. prepare the data


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


#@_catch_error
def predict(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    #if (not any([kwargs['img_style'], kwargs['style']]) or
    #        all([kwargs['img_style'], kwargs['style']])):
    #    return "ERROR : You must provide either custom 'img_style' or choose any of the styles in the 'style' list."

    if kwargs['img_content']:
        return _predict_data(kwargs)
    else:
         return "ERROR : You must provide an image as the content"
    
def _predict_data(args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    message = { "status": "ok",
               "prediction": [],
              }

    # select wether cpu or gpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO]: Running in device: {}".format(device))
    
    # image style.
    img_style_path = args["img_style"].filename
    
    # image content.
    img_content_path = args["img_content"].filename
    
    #img_style = os.path.join(cfg.IMG_STYLE_DIR, 'picasso.jpg')
    img_style = img_style_path
    
    # image content.
    #img_content = os.path.join(cfg.IMG_STYLE_DIR, 'dancing.jpg')
    img_content = img_content_path
    img_content_size = Image.open(img_content)
    width, height = img_content_size.size
    

    print("[INFO]: Resizing images...")
    # image resizing value.
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu 
        
    # convert the image into a torch tensor.
    img_style = iutils.image_loader(img_style, imsize, height, width, device)
    img_content = iutils.image_loader(img_content, imsize, height, width, device)
    
    print("[DEBUG]: Style image size: {}".format(img_style.size()))
    print("[DEBUG]: Content image size: {}".format(img_content.size()))
    
    assert img_style.size() == img_content.size()
    
    # defining VGG-19 layers to get features for styling and content.
    
    print("[INFO]: Defining layers...")
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_layers = ['conv_4']
    
    # VGG networks are trained on images with each channel normalized.
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
    print("[INFO]: Downloading VGG-19 network")
    # load VGG-19 model.
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    # cloning the content image as the input image.
    img_input= img_content.clone()
    
    #A white noise image can also be used as input but it will require more number of steps.
    #img_input = torch.randn(img_content.data.size(), device=device)

    print("[INFO]: Transferring style to image..")
    # run style transfer.
    output = transfer_style.run_style_transfer(cnn, device, normalization_mean, normalization_std,
                            img_content, img_style, img_input, style_layers, content_layers, args["num_steps"],
                                               args["style_weight"], args["content_weight"])
    
    print("[INFO]: Saving image...")
    # saving the image.
    unloader = transforms.ToPILImage() 
    
    # remove the fake batch dimension.
    image = output.cpu().clone()
    image = image.squeeze(0)      # remove the fake batch dimension
    
    img_result = unloader(image)
    img_result.save(os.path.join(cfg.DATA_DIR, 'image_result.png'))
    
    if(args['accept'] == 'image/png'):
        message = open(os.path.join(cfg.DATA_DIR, 'image_result.png'), 'rb')
    
    else:
        prediction_results = {"DONE": "succesfully transferred."}
        message["prediction"].append(prediction_results)
        
    print("[INFO]: Transferring finished.")
    
    return message


def _predict_url(args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


###
# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
###
@flaat.login_required() # Allows only authorized people to train
def train(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    message = { "status": "ok",
                "training": [],
              }

    # use the schema.
    #schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments.
    #train_args = schema.load(kwargs)
    
    # 1. implement your training here. 

    # 2. update "message"
    train_results = { "DONE": "Training is not implemented. Everything is done in 'Prediction'" }
    message["training"].append(train_results)

    return message


# during development it might be practical 
# to check your code from CLI (command line interface)
def main():
    """
    Runs above-described methods from CLI
    (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta      
    elif args.method == 'predict':
        # [!] you may need to take special care in the case of args.files [!]
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'train':
        results = train(**vars(args))
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)

    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"deep_api.py method --help\" to get more info', 
                            dest='method')

    ## configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])                                      
    # normally there are no arguments to configure for get_metadata()

    ## configure arguments for predict()
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser]) 
    # one should convert get_predict_args() to add them in predict_parser
    # For example:
    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    ## configure arguments for train()
    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser]) 
    # one should convert get_train_args() to add them in train_parser
    # For example:
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])

    args = cmd_parser.parse_args()
    
    main()
