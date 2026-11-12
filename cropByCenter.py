
## Code cribbed from https://stackoverflow.com/questions/72851576/corner-detection-in-opencv

import cv2
import subprocess
import argparse
import os

#template_file = './test2.png'
#HIGHPASS = 200 ## spot the very white bits

## Check if file is in pi/date range
## select correct template

def build_parse():
    parser = argparse.ArgumentParser(description='Required and additional inputs')
    parser.add_argument('--in_file','-i',required=True,help='Path to input img or video')
    parser.add_argument('--out_file','-o',required=False,help='Path to output, if not specified, goes to in_file.crop.mp4')
    parser.add_argument('--pad_log','-l',default='./pad_log.txt',required=False,help='Path to log of padded files, if not specifies defaults to ./pad_log.txt')
    parser.add_argument('--template','-t',required=False,help='Path to output, if not specified, goes to in_file.crop.mp4')
    parser.add_argument('--frame','-f',required=False,help='frame to check, defautls to 3000')
    parser.add_argument('--visualize','-v',action='store_true',help='Option to visualize the output')
    parser.add_argument('--dump','-d',action='store_true',help='Option to dump the output, useful for debug')
    return parser.parse_args()


args = build_parse()
in_file = args.in_file

if args.out_file is None:
    out_file = in_file.replace('.mp4','.crop.mp4')
else:
    out_file = args.out_file

f_base = in_file.split('/')[-1]
pi = f_base.split('.')[0]
year = f_base.split('.')[1]
month = f_base.split('.')[2]
day = f_base.split('.')[3]

if args.template is not None:
    template_file = args.template
else:
    template_file = '/home/ammon/Documents/Scripts/FindYourCenter/templateWhiteSimple.png'
#template_file = './templateWhiteSmall.png'
    #magnets = False
    if pi in ['pi09','pi11','pi12']:
        if year == '2024':
            template_file = '/home/ammon/Documents/Scripts/FindYourCenter/templateBlackSimple.png'
            #magnets = True

#template_file = './template.png'
#template_file = './templateBlack.png'
#template_file = './templateBlackSimple.png'
#template_file = './templateWhiteSimple2.png'
#template_file = './templateBlackSquares.png'
#template_file = './templateOneBlack.png'
#template_file = './templateSmallBlack.png'
#template_file = './templateGrayCross.png'
#template_file = './templateGrayBlack.png'
#template_file = './templateNoCenter.png'
HIGHPASS = 70   ## Spot the black bits

## Grab image from video
cap = cv2.VideoCapture(in_file)
if args.frame is None:
    START_FRAME = 3000
else:
    START_FRAME = int(args.frame)
cap.set(cv2.CAP_PROP_POS_FRAMES,START_FRAME) ## need to skip forward to avoid early black stuff
#import pdb;pdb.set_trace()
res,image = cap.read()

cap.release()

base_dims = image.shape[:2]
#image = cv2.imread(filename)

image = cv2.copyMakeBorder(image,300,300,0,0,cv2.BORDER_CONSTANT)
template = cv2.imread(template_file,cv2.IMREAD_UNCHANGED)

tmplt_mask = template[:,:,3]
tmplt_mask = cv2.merge([tmplt_mask,tmplt_mask,tmplt_mask])

template = template[:,:,0:3]

if True:
    image = cv2.bitwise_not(image)
    template = cv2.bitwise_not(template)

gray = image
if False:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image[:,:,2]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

if False:
    gray[gray < HIGHPASS] = 0
    gray[gray > HIGHPASS] = 255

blur = cv2.GaussianBlur(gray,(11,11),0)
blur = cv2.GaussianBlur(blur,(11,11),0)

if False:
    blur[0:200] = 0
    blur[800:] = 0
    blur[:,0:200] = 0
    blur[:,800:] = 0

#from matplotlib import pyplot as plt
#template = blur[500:600,420:520]
#plt.imsave('test',template,cmap='bw')

w_i,h_i = gray.shape[1],gray.shape[0]
w,h = template.shape[1],template.shape[0]
#w, h = template.shape[(1,0)]
#detected = cv2.matchTemplate(blur,template,cv2.TM_CCOEFF)
detected = cv2.matchTemplate(blur,template,cv2.TM_CCORR_NORMED,mask=tmplt_mask)

#import pdb;pdb.set_trace()
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(detected)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

top_left = max_loc
#center = top_left + 60
center = (top_left[0] + int(w/2), top_left[1] + int(h/2))
bottom_right = (top_left[0] + w, top_left[1] + h)

x_shift = 440
y_shift = 515

x_min = max(0,center[0] - x_shift)
x_max = min(w_i,center[0] + x_shift)
y_min = max(0,center[1] - y_shift)
y_max = min(h_i,center[1] + y_shift)


pad_h = h_i
pad_w = w_i
crop_w = x_max - x_min
crop_h = y_max - y_min
tmp_file = in_file.replace('mp4','tmp.mp4')

if args.visualize:
    match_image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.rectangle(image,top_left,bottom_right,255,2)
    cv2.rectangle(image,(x_min,y_min),(x_max,y_max),255,2)
    cv2.circle(image,(center[0],center[1]),5,(0,255,0),-1)
    import matplotlib.pyplot as plt

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

    axes[0].imshow(image_rgb)
    axes[0].set_title(f'Detected center (match={max_val:.4f})')
    axes[0].axis('off')

    axes[1].imshow(image_rgb)
    heatmap = axes[1].imshow(
        detected,
        cmap='inferno',
        alpha=0.65,
        origin='upper',
        extent=(0, detected.shape[1], detected.shape[0], 0),
    )
    axes[1].scatter(
        max_loc[0], max_loc[1],
        marker='x', s=90, linewidths=2, color='cyan', label='Best match top-left'
    )
    axes[1].set_xlim(0, image.shape[1])
    axes[1].set_ylim(image.shape[0], 0)
    axes[1].set_title('Template-match response over valid top-left positions')
    axes[1].legend(loc='upper right')
    fig.colorbar(heatmap, ax=axes[1], label='TM_CCORR_NORMED score')

    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template_overlay = match_image_rgb.copy()
    match_x, match_y = top_left
    overlay_region = template_overlay[match_y:match_y + h, match_x:match_x + w]
    alpha = tmplt_mask[:, :, :1].astype(float) / 255.0
    overlay_region[:] = (
        overlay_region.astype(float) * (1.0 - alpha)
        + template_rgb.astype(float) * alpha
    ).astype('uint8')
    axes[2].imshow(template_overlay)
    axes[2].add_patch(
        plt.Rectangle(top_left, w, h, fill=False, edgecolor='cyan', linewidth=2)
    )
    axes[2].set_title('Template overlaid at best match')
    axes[2].axis('off')
    plt.show()

if args.dump:
    print('top left:',x_min,y_min)
    print('center:',center[0],center[1])
    print('did I do it?')
    #import pdb;pdb.set_trace()
    pass
elif y_min - 300 < 0  or y_max - 300 > base_dims[0]:
    ## pad by 300
    ## ffmpeg pad then crop
    x0 = x_min
    y0 = y_min
    print('padding and cropping. This takes a minute')
    #cmd = f"ffmpeg -i {in_file} -vf 'pad={pad_w}:{pad_h}:0:300:color=black' {tmp_file} -y -hide_banner -loglevel error"
    #cmd2 = f"ffmpeg -i {tmp_file} -vf 'crop={crop_w}:{crop_h}:{x0}:{y0}' {out_file} -y -hide_banner -loglevel error"
## single command to both pad and crop
    cmd = f"ffmpeg -i {in_file} -threads 8 -vf 'pad={pad_w}:{pad_h}:0:300:color=black,crop={crop_w}:{crop_h}:{x0}:{y0}' {out_file} -y -hide_banner -loglevel error"
    print('added to list of padded vids',args.pad_log)
    with open(args.pad_log,'a') as f:
        f.write(out_file + '\n')
    subprocess.call(cmd,shell=True)

    #subprocess.call(cmd2,shell=True)
    #os.remove(tmp_file)
    #print(cmd)
else:
    y0 = y_min - 300
    x0 = x_min
    w = x_max - x_min
    h = y_max - y_min
    print('Cropping. This takes a minute')
    cmd2 = f"ffmpeg -i {in_file} -threads 8 -vf 'crop={crop_w}:{crop_h}:{x0}:{y0}' {out_file} -y -hide_banner -loglevel error"
    subprocess.call(cmd2,shell=True)
