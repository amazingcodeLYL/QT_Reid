import argparse
import time
from sys import platform

from numpy import imag
from parser_detect import *
from models import *
from ui_yolov3.datasets import *
from utils.utils import *


def detect(cfg,
           data,
           weights,
           images=None,  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           webcam=False,
           img_size=416,
           conf_thres=0.2,
           nms_thres=0.2,
           save_txt=False,
           save_images=True):

    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # 删除output文件夹，清理之前的检测结果
    os.makedirs(output)        # 创建新的output文件夹

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        model.half()

    classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # 对于每种类别随机使用一种颜色画框
    
    im0 = images.copy() 
    #  #im0.shape (480, 640, 3)
    # im0 = im0.transpose(2,0,1)
    # print("im0.shape",im0.shape)


    images = torch.from_numpy(images).permute(2,0,1).unsqueeze(0).to(device)
    images = images.float()
    images = images.to(device)
        
    print(images.shape)
    pred, _ = model(images) # 经过处理的网络预测，和原始的
    # print(pred)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0] # torch.Size([5, 7])
    # print(det)
    if det is not None and len(det) > 0:
        print("det",det)
        # Rescale boxes from 416 to true image size 映射到原图
        det[:, :4] = scale_coords(images.shape[2:], det[:, :4], im0.shape).round()

        # Print results to screen image 1/3 data\samples\000493.jpg: 288x416 5 persons, Done. (0.869s)
        # print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
        for c in det[:, -1].unique():   # 对图片的所有类进行遍历循环
            n = (det[:, -1] == c).sum() # 得到了当前类别的个数，也可以用来统计数目
            # print('%g %ss' % (n, classes[int(c)]), end=', ') # 打印个数和类别'5 persons'

        # Draw bounding boxes and labels of detections
        # (x1y1x2y2, obj_conf, class_conf, class_pred)
        count = 0
        for *xyxy, conf, cls_conf, cls in det: # 对于最后的预测框进行遍历
            print("对于最后的预测框进行遍历")
            # *xyxy: 对于原图来说的左上角右下角坐标: [tensor(349.), tensor(26.), tensor(468.), tensor(341.)]
            # if save_txt:  # Write to file
            #     with open(save_path + '.txt', 'a') as file:
            #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf) # 'person 1.00'
            print("label",label)
            # 只显示检测的人
            if classes[int(cls)] == 'person':
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                print("plot_one_box")
    return im0



# if __name__ == '__main__':
def main_detect(images_source):

    with torch.no_grad():
        im0 = detect(opt.cfg,
               opt.data,
               opt.weights,
               images=images_source,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
    return im0