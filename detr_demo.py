import torch as t
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import requests

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    # T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_image_url(url):
    img = Image.open(requests.get(url, stream=True).raw).resize((800, 600))
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def load_image_local(image):
    img = Image.open(image).resize((800,600))
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def predict(model, img_tensor):
    with t.no_grad():
        output = model(img_tensor)
    pred_logits = output['pred_logits'][0]
    pred_boxes = output['pred_boxes'][0]
    return pred_logits, pred_boxes


def plot_results(image,logits,boxes,fontsize):
    drw = ImageDraw.Draw(image)
    count = 0
    for logit, box in zip(logits, boxes):
        cls = logit.argmax()
        if cls >= len(CLASSES):  # if the class is larger than the length of CLASSES, we will just skip for now
            continue
        count += 1
        label = CLASSES[cls]
        box = box * t.Tensor([800, 600, 800, 600])  # scale up the box to the original size
        x, y, w, h = box
        x0, x1 = x - w//2, x + w//2
        y0, y1 = y - h//2, y + h//2
        print('object {}: label:{},box:{}'.format(count,label,box))  # [x,y,w,h]
        drw.rectangle([x0,y0,x1,y1], outline='red',width=1)
        font = ImageFont.truetype("consola.ttf", fontsize, encoding="unic")
        drw.text((x,y), label, 'blue', font)
    print('{} objects found'.format(count))


if __name__ == '__main__':
    model = t.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
    model.eval()

    # url = input()  # use an arbitrary online image
    # url = 'http://images.cocodataset.org/val2017/000000012120.jpg'
    # img, img_tensor = load_image_url(url)
    image = 'test.jpg'
    img, img_tensor = load_image_local(image)

    pred_logits, pred_boxes = predict(model, img_tensor)

    img_cp = img.copy()
    plot_results(img_cp, pred_logits, pred_boxes, 15)
    img_cp.show()
