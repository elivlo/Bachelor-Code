from datetime import datetime
import layoutparser as lp
import cv2
import os
import json
import random

test_sites_list = [40001, 40002, 40005, 40006, 40007, 40008, 40009, 40010, 40011,
                   40012, 40013, 40014, 40015, 40016, 40017, 40018, 40019, 40020, 40021,
                   40022, 40023, 40024, 40025, 40026, 40027, 40028, 40029, 40030, 40031,
                   40032, 40033, 40034, 40035, 40036, 40037, 40038, 40039, 40040,
                   40042, 40043, 40044]

base_path = "../phishing_websites/legit_database_files"

#model_pub = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
#                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
#                                     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

model_prima = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                       extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
                                       label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})


def is_subinterval(start1, end1, start2, end2):
    """1 is subinterval of 2"""
    return int(start1) >= int(start2) and int(end1) <= int(end2)


def rect_in_rect(rect1, rect2, margin=0):
    # rect2 = rect2.pad(margin,margin,margin,margin)
    r1_x1, r1_y1, r1_x2, r1_y2 = rect1.coordinates
    r2_x1, r2_y1, r2_x2, r2_y2 = rect2.coordinates
    return (
        is_subinterval(r1_x1, r1_x2, r2_x1, r2_x2) and
        is_subinterval(r1_y2, r1_y1, r2_y2, r2_y1)
    )


def is_in(e, layout):
    for i, elem in enumerate(layout):
        if e is elem:
            continue
        if rect_in_rect(e, elem, 8):
            return True, -1
        if rect_in_rect(elem, e, 8):
            return False, i
    return False, -1


def remove_is_in(layout, center=True):
    lay = lp.Layout()

    for element in layout:
        inside = False
        for e in lay:
            if e is element:
                inside = True
                break
            if element.block.x_1 >= e.block.x_1 and element.block.y_1 >= e.block.y_1 and element.block.x_2 <= e.block.x_2 and element.block.y_2 <= e.block.y_2:
                inside = True
                break
        if not inside:
            lay += [element]
    return lay


def remove_big_small(layout, img_size):
    lay = lp.Layout()
    for element in layout:
        area = (element.area / img_size)
        if 0.5 > area > 0.05:
            lay += [element]
    return lay


def is_in_line(layout, im_height):
    lay = lp.Layout()

    for e in layout:
        y_center = e.block.center[1]
        found = False
        for elem in lay:
            if abs(y_center - elem.block.center[1]) < (im_height*0.05):
                found = True
                if e.block.x_1 < elem.block.x_1:
                    elem.block.x_1 = e.block.x_1
                if e.block.x_2 > elem.block.x_2:
                    elem.block.x_2 = e.block.x_2
                if e.block.y_1 < elem.block.y_1:
                    elem.block.y_1 = e.block.y_1
                if e.block.y_2 > elem.block.y_2:
                    elem.block.y_2 = e.block.y_2
        if not found:
            lay.insert(0, e)
    return lay.sort(key=lambda x: x.area, reverse=True)


def is_in_row(layout, im_width):
    lay = lp.Layout()

    for e in layout:
        x_center = e.block.center[0]
        found = False
        for elem in lay:
            if abs(x_center - elem.block.center[0]) < (im_width*0.05):
                found = True
                if e.block.y_1 < elem.block.y_1:
                    elem.block.y_1 = e.block.y_1
                if e.block.y_2 > elem.block.y_2:
                    elem.block.y_2 = e.block.y_2
                if e.block.x_1 < elem.block.x_1:
                    elem.block.x_1 = e.block.x_1
                if e.block.x_2 > elem.block.x_2:
                    elem.block.x_2 = e.block.x_2
        if not found:
            lay.insert(0, e)
    return lay.sort(key=lambda x: x.area, reverse=True)


def own_images():
    # ZUM ZEIGEN: Die verschiedenen Stufen zeigen, damit bewiesen werden kann,
    # dass das Layout angepasst werden muss, damit viele gute Bereiche gefunden werden
    for n in test_sites_list:
        image = cv2.imread(f"{base_path}/{n}/screenshot.png")
        image = image[..., ::-1]
        image_size = image.shape[0] * image.shape[1]

        start = datetime.now()

        lay = model_prima.detect(image)

        lines = is_in_line(lay.sort(key=lambda x: x.block.center[1]), image.shape[0])
        rows = is_in_row(lay.sort(key=lambda x: x.block.center[0]), image.shape[1])

        l = remove_big_small(lines, image_size)
        r = remove_big_small(rows, image_size)
        lr = remove_is_in((l + r).sort(key=lambda x: x.area, reverse=True))

        end = datetime.now()
        print(f"Runtime: {(end - start).total_seconds() * 1000}")
        if len(lr) > 0:
            print(lr.to_dataframe().loc[:, ["x_1", "x_2", "y_1", "y_2"]].to_numpy())


        image_begin = lp.draw_box(image, lay, box_width=3)
        image_end = lp.draw_box(image, lr, box_width=3)
        print(n)
        print("##############################################################")

        image_begin.save(f"publaynet{n}.png")
        image_end.save(f"publaynet{n}_2.png")


def split_coco():
    test_data = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": 1, "name": "box"}]
    }
    train_data = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": 1, "name": "box"}]
    }

    with open("coco_set.json", "r") as infile:
        data = json.load(infile)
        random.shuffle(data["images"])

        arr_train_len = int(len(data["images"]) * 0.95)

        for i, im in enumerate(data["images"]):
            if i < arr_train_len:
                train_data["images"].append(im)
            else:
                test_data["images"].append(im)

            anno_list = []
            im_id = im["id"]
            for anno in data["annotations"]:
                if anno["image_id"] == im_id:
                    anno_list.append(anno)

            if i < arr_train_len:
                train_data["annotations"].extend(anno_list)
            else:
                test_data["annotations"].extend(anno_list)

        print(f'Test: im {len(test_data["images"])} anno {len(test_data["annotations"])}')
        print(f'Train: im {len(train_data["images"])} anno {len(train_data["annotations"])}')

        with open("coco_set_test.json", "w") as outfile:
            json.dump(test_data, outfile)

        with open("coco_set_train.json", "w") as outfile:
            json.dump(train_data, outfile)


def main():
    benign_images_test = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": 1, "name": "box"}]
    }

    counter = 0
    area_counter = 0
    for filename in os.listdir("../benign_sample_30k/"):
        print(counter)
        image_filename = f"{filename}/shot.png"

        image = cv2.imread(f"../benign_sample_30k/{image_filename}")
        if image is None:
            continue
        image = image[..., ::-1]
        height, width, _ = image.shape
        image_size = height * width

        layout_pub = model_pub.detect(image).pad(8, 8, 8, 8)
        layout_prima = model_prima.detect(image).pad(8, 8, 8, 8)
        lay = (layout_pub + layout_prima).sort(key=lambda x: x.area)

        lines = is_in_line(lay)
        rows = is_in_row(lay)

        l = remove_is_in(lines)
        r = remove_is_in(rows)
        lr = remove_big((l + r), image_size)

        im = {
            "file_name": image_filename,
            "height":    height,
            "width":     width,
            "id":        counter
        }
        benign_images_test["images"].append(im)

        for elem in lr:
            x, y, _, _ = elem.coordinates
            width = int(elem.width)
            height = int(elem.height)

            area = {
                "area":        width * height,
                "image_id":    counter,
                "bbox":        [int(x), int(y), width, height],
                "category_id": 1,
                "id":          area_counter,
                "iscrowd":     0
            }
            benign_images_test["annotations"].append(area)
            area_counter += 1

        counter += 1

    with open("coco_set.json", "w") as outfile:
        json.dump(benign_images_test, outfile)


if __name__ == "__main__":
    own_images()
    #main()
    #split_coco()
