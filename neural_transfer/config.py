# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields, validate
from marshmallow import Schema, INCLUDE

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
IMG_STYLE_DIR = os.path.join(IN_OUT_BASE_DIR, 'neural_transfer/dataset/style_images')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')

# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    
    img_content = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="image_content",
        location="form",
        description="Image to be styled."
    )
    
    img_style = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="image_style",
        location="form",
        description="Image with the style."
    )
    
    style = fields.Str(
            required=False,  # force the user to define the value
            enum=["The Starry Night - Van Gogh" , "Mosaic Lady", "Seated Nude - Picasso", "The Great Wave off Kanagawa - Hokusai"],  # list of choices
            description="Selection of the image which style we want to transfer. Select one if you don't have any."  # help string
        )
    
    num_steps = fields.Int(
        required=False,
        missing = 300,
        description="Number of iterations on the network to compute the gradients."
    )
      
    style_weight =  fields.Int(
        required=False,
        missing = 1000000,
        description="Weigth of the image of the style. It represents the emphasis on style in the image. There is a tradeoff between style weight and content weight."
    )
     
    content_weight =  fields.Float(
        required=False,
        missing = 1,
        description="Weigth of the image of the content. It represents the emphasis on content in the image. There is a tradeoff between style weight and content weight"
    )
        
    accept = fields.Str(
            require=False,
            description="Returns the image with the new style or a pdf containing the 3 images.",
            missing='image/png',
            validate=validate.OneOf(['image/png', 'application/pdf']))
    
# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    #name = fields.Str(
    #    required=True,
    #    location="form",
    #    description="Description"
    #)
