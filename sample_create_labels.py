import os
from PIL import Image, ImageDraw


img_dir = './images'
label_dir = './labels'
save_dir = './out_labels'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
img_w, img_h = (640, 480)

idx = 1
for filename in os.listdir(label_dir):
    print('Label count = ',idx)
    idx += 1
    labels_per_file = []
    with open(os.path.join(label_dir, filename)) as fp:
        for line in fp.readlines():
            label = line.strip().split(' ')
            xyxy = [float(label[4]) / img_w, float(label[5]) / img_h, float(label[6]) / img_w, float(label[7]) / img_h]
            xyxy = [min(max(x, 0), 1) for x in xyxy]

            xc = (xyxy[0] + xyxy[2]) / 2
            yc = (xyxy[1] + xyxy[3]) / 2
            
            w = xyxy[2] - xyxy[0] 
            h = xyxy[3] - xyxy[1] 
            cls = 1 if 'no' in label[0].lower() else 0
            
            label = [str(cls)] + [str(xc), str(yc), str(w), str(h)]
            labels_per_file.append(' '.join(label))

    with open(os.path.join(save_dir, filename), 'w') as fp:
        fp.write('\n'.join(labels_per_file))
            # draw.rectangle(((xyxy[0] * img_w, xyxy[1] * img_h), (xyxy[2] * img_w, xyxy[3] * img_h)))
# img.show()
