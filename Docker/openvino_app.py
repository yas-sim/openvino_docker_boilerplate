import cv2
import numpy as np
import openvino as ov

input_image = './input/image.jpg'
output_image = './output/image.jpg'
#input_image = '../input/image.jpg'    ## for test
#output_image = '../output/image.jpg'  ## for test
model_file = './ssd_mobilenet_v1_coco_2018_01_28.xml'
coco_label_file = './coco_label.txt'

# Draw label text on an image
def draw_label(img, x, y, text, color_fg=(0,0,0), color_bg=(0,255,0)):
    face = cv2.FONT_HERSHEY_PLAIN
    scale = 1
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, face, scale, thickness)
    cv2.rectangle(img, (x, y - height - baseline), (x + width, y), color_bg, -1)
    cv2.putText(img, text, (x, y - baseline), face, scale, color_fg, thickness)


def main():
    with open(coco_label_file, 'rt') as f:
        coco_labels = [ file.rstrip('\n') for file in f.readlines() ]
    #print(coco_labels)

    ov_model = ov.Core().read_model(model_file)
    #print(ov_model.outputs)
    output_det_box = 'detection_boxes:0'
    output_det_cls = 'detection_classes:0'
    output_det_scr = 'detection_scores:0'
    output_det_num = 'num_detections:0'

    compiled_model = ov.compile_model(ov_model, device_name='CPU')

    # Prepare the input image
    img_size = 300
    img = cv2.imread(input_image)
    img = cv2.resize(img, (img_size, img_size))
    result_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[np.newaxis, :, :, :]
    #img = img / 255

    # Run inference
    res = compiled_model(img)

    num_det = int(res[output_det_num][0])
    print(f'{num_det} objects detected.')

    # Draw bounding boxes and labels
    threshold = 0.7
    for n in range(num_det):
        box = res[output_det_box][0][n]
        cls = res[output_det_cls][0][n]
        scr = res[output_det_scr][0][n]
        print(box, cls, scr)
        if scr >= threshold:
            y0 = int(box[0] * img_size)
            x0 = int(box[1] * img_size)
            y1 = int(box[2] * img_size)
            x1 = int(box[3] * img_size)
            cls = int(cls) - 1
            cv2.rectangle(result_img, (x0, y0), (x1, y1), (0,255,0), 2)
            draw_label(result_img, x0, y0, coco_labels[cls])

    # Write the final result image to a file
    cv2.imwrite(output_image, result_img)

if __name__ == '__main__':
    main()
