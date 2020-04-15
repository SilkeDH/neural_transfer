import neural_transfer.config as cfg
import os
from PIL import Image
from fpdf import FPDF
        
        
# Input and result images from the segmentation
files = ['{}/content_image.png'.format(cfg.DATA_DIR),
         '{}/style_image.png'.format(cfg.DATA_DIR),
         '{}/result_image.png'.format(cfg.DATA_DIR)]

# Merge images and add a color legend
def merge_images():
    for path in files:
        img = Image.open(path)
        img.thumbnail((310, 210), Image.ANTIALIAS)
        img.save(path)


# Put images and accuracy information together in one pdf file
def create_pdf(style_score, content_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Results:', ln=1)
    pdf.set_font('Arial', '',12)
    pdf.cell(0, 10, 'Applying style:', ln=1)
    pdf.image(files[1], 10, 30)
    
    pdf.cell(0, 170, 'To image:', ln=1)
    pdf.image(files[0], 10, 120)
    
    pdf.cell(0, 20, 'With:', ln=1) 
    pdf.cell(0, 5, '\t\t\t\tStyle loss: \t\t {}'.format(style_score), ln=1)
    pdf.cell(0, 10, '\t\t\t\tContent loss: \t\t {}'.format(content_score), ln=1)
    pdf.set_font('Arial', 'I', size=9)
    pdf.cell(1, 30, '--> Result image is in the next page.', ln=2)
    
    pdf.add_page()
    pdf.set_font('Arial', '',12)
    pdf.cell(0, 10, 'Result image:', ln=1) 
    pdf.image(files[2], 10, 30)
    results = '{}/prediction_results.pdf'.format(cfg.DATA_DIR)
    pdf.output(results,'F')
    return results